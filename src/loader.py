import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import random
from tqdm import tqdm

# Dataset 클래스 정의
class EHRDataset(Dataset):
    def __init__(self, data):
        self.data = []
        self.patient_ids = []
        self.visit_lengths = []
        self.labels = []
        self.time_deltas = []
        # 각 환자별로 모든 방문을 하나의 샘플로 저장
        for patient_id, patient_data in data.items():
            self.patient_ids.append(patient_id)  # 환자 ID 저장
            self.data.append([torch.tensor(visit) for visit in patient_data['seq_idx']])  # 방문 기록을 텐서로 저장
            self.visit_lengths.append(patient_data['visit_length'])
            self.time_deltas.append(torch.tensor(patient_data['timedelta'], dtype=torch.long))
            self.labels.append(patient_data['label'])  # 환자의 라벨 저장

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.patient_ids[idx], self.data[idx], self.visit_lengths[idx], self.time_deltas[idx], self.labels[idx]

# 패딩 값을 인자로 받는 collate_fn 정의
def collate_fn(batch, padding_value=0):
    patient_id = [item[0] for item in batch]  # 환자 ID 추출
    visit_data = [item[1] for item in batch]  # 방문 기록만 추출 (각 환자의 방문 리스트)
    visit_lengths = [item[2] for item in batch]  # 방문 길이 추출
    time_deltas = [item[3] for item in batch]
    labels = [torch.tensor(item[4], dtype=torch.float32) for item in batch]  # 라벨 추출
    
    # 각 방문 내 시퀀스(진단 코드 리스트)를 먼저 패딩하여 동일한 길이로 만듦
    max_code_length = max(max(len(visit) for visit in visits) for visits in visit_data)
    padded_visits = [
        [torch.cat([visit, torch.full((max_code_length - len(visit),), padding_value)]) for visit in visits]
        for visits in visit_data
    ]
    
    # 환자의 방문 리스트에도 패딩을 적용하여 방문 횟수도 맞춤
    max_num_visits = max(len(visits) for visits in padded_visits)
    for i in range(len(padded_visits)):
        while len(padded_visits[i]) < max_num_visits:
            padded_visits[i].append(torch.full((max_code_length,), padding_value))
            
    padded_tds = [
        torch.cat([td, torch.full((max_num_visits - len(td),), 100000)]) for td in time_deltas
    ]
    
    # 텐서를 3D로 변환
    patient_id = torch.tensor(patient_id)
    padded_visits_seq = torch.stack([torch.stack(visits) for visits in padded_visits])
    visit_lengths = torch.tensor(visit_lengths, dtype=torch.long)
    padded_tds = torch.stack(padded_tds)
    seq_mask = (padded_tds != 100000).float()
    final_index = torch.where(seq_mask == 1)[0].max().item()
    seq_mask_final = torch.zeros_like(seq_mask)
    seq_mask_final[final_index] = 1
    
    seq_mask_code = (padded_visits_seq != 0).float()
    labels = torch.stack(labels)
    
    return {'patient_id': patient_id, 
            'visit_seq': padded_visits_seq, 
            'length': visit_lengths, 
            'time_delta': padded_tds,
            'seq_mask': seq_mask,
            'seq_mask_final': seq_mask_final, 
            'seq_mask_code': seq_mask_code, 
            'labels': labels}
    
    

class PretrainedEHRDataset(Dataset):
    def __init__(self, data, mask_prob=0.15, mask_idx=5091, max_pred=10, max_len=400, padding_value=0):
        self.mask_prob = mask_prob
        self.mask_idx = mask_idx
        self.max_pred = max_pred
        self.data_ids = []
        self.data = []
        self.masked_data = []
        self.code_type = []
        self.visit_seg = []
        self.seq_len = []
        self.time_deltas = []
        self.times = []
        self.mask_tokens = []
        self.mask_pos = []
        self.labels = []
        
        # self.seq_len = [d['total_len'] for _, d in data.items()]
        # max_seq_len = max(self.seq_len)
        max_seq_len = max_len
        # 각 환자별로 모든 방문을 하나의 샘플로 저장
        idx = 0
        for patient_id, patient_data in tqdm(data.items()):
            p_id, num = patient_id.split('_slide')
            self.data_ids.append(torch.tensor([int(p_id), int(num)]))  # 환자 ID 저장
            visit_seq = torch.tensor(patient_data['total_seq_idx'])
            code_type = torch.tensor(patient_data['code_type'])
            masked_seq, mask_tokens, mask_pos = self.masked_code(visit_seq, code_type)
            visit_seg = torch.tensor(patient_data['visit_seg'])
            timedelta = torch.tensor(patient_data['timedelta'], dtype=torch.long)
            timevec = torch.tensor(patient_data['timevec'], dtype=torch.long)
            seq_len = patient_data['total_len']
            self.seq_len.append(seq_len)
            self.data.append(F.pad(visit_seq, (0, max_seq_len - seq_len), "constant", padding_value))  # 방문 기록을 텐서로 저장
            masked_seq = F.pad(masked_seq, (0, max_seq_len - seq_len), "constant", padding_value)
            self.masked_data.append(masked_seq)
            self.visit_seg.append(F.pad(visit_seg, (0, max_seq_len - seq_len), "constant", 50))
            self.code_type.append(F.pad(code_type, (0, max_seq_len - seq_len), "constant", padding_value))
            self.time_deltas.append(F.pad(timedelta, (0, max_seq_len - seq_len), "constant", 100000))
            self.times.append(F.pad(timevec, (0, 0, 0, max_seq_len - seq_len), "constant", 0))
            self.mask_tokens.append(mask_tokens)
            self.mask_pos.append(mask_pos)
            self.labels.append(torch.tensor(patient_data['label'], dtype=torch.float32))  # 환자의 라벨 저장
            idx += 1
        
    def masked_code(self, seq_data, code_types):
        type_1_indices = [i for i, token_type in enumerate(code_types) if token_type == 1]
        num_masks = min(self.max_pred, int(len(type_1_indices) * self.mask_prob))
        masked_indices = random.sample(type_1_indices, num_masks)
        masked_sequence = seq_data.clone()
        masked_tokens = F.pad(masked_sequence[masked_indices], (0, self.max_pred - len(masked_indices)), "constant", 0)
        masked_pos = F.pad(torch.tensor(masked_indices), (0, self.max_pred - len(masked_indices)), "constant", 0)
        masked_sequence[masked_indices] = self.mask_idx
        return masked_sequence, masked_tokens, masked_pos

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (self.data_ids[idx],     # 0
                self.data[idx],         # 1
                self.masked_data[idx],  # 2
                self.visit_seg[idx],    # 3
                self.code_type[idx],    # 4
                self.seq_len[idx],      # 5
                self.time_deltas[idx],  # 6
                self.times[idx],        # 7
                self.mask_tokens[idx],  # 8
                self.mask_pos[idx],     # 9
                self.labels[idx])       # 10


# # 패딩 값을 인자로 받는 collate_fn 정의
def collate_fn_pt(batch):
    data_idx = [item[0] for item in batch]  # 환자 ID 추출
    visit_data = [item[1] for item in batch]  # 방문 기록만 추출 (각 환자의 방문 리스트)
    masked_visit_data = [item[2] for item in batch]  # 방문 기록만 추출 (각 환자의 방문 리스트)
    visit_seg = [item[3] for item in batch]  # 방문 순서 segment
    code_types = [item[4] for item in batch]  # code type
    seq_lengths = [item[5] for item in batch]  # 방문 길이 추출
    time_deltas = [item[6] for item in batch]
    times = [item[7] for item in batch]
    mask_tokens = [item[8] for item in batch]
    mask_pos = [item[9] for item in batch]
    labels = [item[10] for item in batch]  # 라벨 추출

    # 텐서를 3D로 변환
    data_ids = torch.stack(data_idx)
    visit_data = torch.stack(visit_data)
    masked_visit_data = torch.stack(masked_visit_data)
    visit_lengths = torch.tensor(seq_lengths, dtype=torch.long)
    visit_seg = torch.stack(visit_seg)
    code_types = torch.stack(code_types)
    time_deltas = torch.stack(time_deltas)
    timevecs = torch.stack(times)
    mask_tokens = torch.stack(mask_tokens)
    mask_pos = torch.stack(mask_pos)
    seq_mask = (time_deltas == 100000).bool()
    labels = torch.stack(labels)
    
    return {'data_idx': data_ids, 
            'visit_seq': visit_data,
            'masked_visit_seq': masked_visit_data,
            'length': visit_lengths, 
            'visit_segments': visit_seg,
            'code_types': code_types,
            'time_delta': time_deltas,
            'timevec': timevecs,
            'seq_mask': seq_mask,
            'mask_tokens': mask_tokens,
            'mask_pos': mask_pos, 
            'labels': labels}









# class PretrainedEHRDataset(Dataset):
#     def __init__(self, data, mask_prob=0.15, mask_idx=5091, max_pred=10):
#         self.mask_prob = mask_prob
#         self.mask_idx = mask_idx
#         self.max_pred = max_pred
#         self.data = []
#         self.masked_data = []
#         self.code_type = []
#         self.visit_seg = []
#         self.data_ids = []
#         self.seq_len = []
#         self.time_deltas = []
#         self.times = []
#         self.labels = []
        
#         # 각 환자별로 모든 방문을 하나의 샘플로 저장
#         for patient_id, patient_data in tqdm(data.items()):
#             p_id, num = patient_id.split('_slide')
#             self.data_ids.append(torch.tensor([int(p_id), int(num)]))  # 환자 ID 저장
#             visit_tensor = torch.tensor(patient_data['total_seq_idx'])
#             code_type_tensor = torch.tensor(patient_data['code_type'])
#             masked_visit_tensor = self.masked_code(visit_tensor, code_type_tensor)
#             self.data.append(visit_tensor)  # 방문 기록을 텐서로 저장
#             self.masked_data.append(masked_visit_tensor)
#             self.visit_seg.append(torch.tensor(patient_data['visit_seg']))
#             self.seq_len.append(patient_data['total_len'])
#             self.code_type.append(code_type_tensor)
#             self.time_deltas.append(torch.tensor(patient_data['timedelta'], dtype=torch.long))
#             self.times.append(torch.tensor(patient_data['timevec'], dtype=torch.long))
#             self.labels.append(patient_data['label'])  # 환자의 라벨 저장
        
#     def masked_code(self, seq_data, code_types):
#         type_1_indices = [i for i, token_type in enumerate(code_types) if token_type == 1]
#         num_masks = min(self.max_pred, int(len(type_1_indices) * self.mask_prob))
#         masked_indices = random.sample(type_1_indices, num_masks)
#         masked_sequence = seq_data.clone()
#         masked_sequence[masked_indices] = self.mask_idx
#         return masked_sequence

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         return (self.data_ids[idx],
#                 self.data[idx],
#                 self.masked_data[idx],
#                 self.visit_seg[idx],
#                 self.code_type[idx],
#                 self.seq_len[idx], 
#                 self.time_deltas[idx], 
#                 self.times[idx],
#                 self.labels[idx])


# # 패딩 값을 인자로 받는 collate_fn 정의
# def collate_fn_pt(batch, padding_value=0):
#     data_idx = [item[0] for item in batch]  # 환자 ID 추출
#     visit_data = [item[1] for item in batch]  # 방문 기록만 추출 (각 환자의 방문 리스트)
#     masked_visit_data = [item[2] for item in batch]  # 방문 기록만 추출 (각 환자의 방문 리스트)
#     visit_seg = [item[3] for item in batch]  # 방문 순서 segment
#     code_types = [item[4] for item in batch]  # code type
#     seq_lengths = [item[5] for item in batch]  # 방문 길이 추출
#     time_deltas = [item[6] for item in batch]
#     times = [item[7] for item in batch]
#     labels = [torch.tensor(item[8], dtype=torch.float32) for item in batch]  # 라벨 추출
    
#     # 각 방문 내 시퀀스(진단 코드 리스트)를 먼저 패딩하여 동일한 길이로 만듦
#     max_seq_length = max(seq_lengths)
#     padded_visits_seq = [
#         F.pad(visit, (0, max_seq_length - len(visit)), "constant", padding_value) for visit in visit_data
#     ]
#     padded_masked_visits_seq = [
#         F.pad(masked_visit, (0, max_seq_length - len(masked_visit)), "constant", padding_value) for masked_visit in masked_visit_data
#     ]
#     padded_vis_seg = [
#         F.pad(seg, (0, max_seq_length - len(seg)), "constant", 50) for seg in visit_seg
#     ]
#     padded_code_type = [
#         F.pad(ct, (0, max_seq_length - len(ct)), "constant", 0) for ct in code_types
#     ]
#     padded_tds = [
#         F.pad(td, (0, max_seq_length - len(td)), "constant", 100000) for td in time_deltas
#     ]
#     padded_tvs = [
#         F.pad(tv, (0, 0, 0, max_seq_length - len(tv)), "constant", 0) for tv in times
#     ]

#     # 텐서를 3D로 변환
#     data_ids = torch.stack(data_idx)
#     padded_visits_seq = torch.stack(padded_visits_seq)
#     padded_masked_visits_seq = torch.stack(padded_masked_visits_seq)
#     visit_lengths = torch.tensor(seq_lengths, dtype=torch.long)
#     padded_vis_seg = torch.stack(padded_vis_seg)
#     padded_code_type = torch.stack(padded_code_type)
#     padded_tds = torch.stack(padded_tds)
#     padded_tvs = torch.stack(padded_tvs)
#     seq_mask = (padded_tds != 100000).bool()
#     mlm_labels = (padded_masked_visits_seq == 5091)
#     labels = torch.stack(labels)
    
#     return {'data_idx': data_ids, 
#             'visit_seq': padded_visits_seq,
#             'masked_visit_seq': padded_masked_visits_seq,
#             'length': visit_lengths, 
#             'visit_segments': padded_vis_seg,
#             'code_types': padded_code_type,
#             'time_delta': padded_tds,
#             'seq_mask': seq_mask,
#             'mlm_labels': mlm_labels, 
#             'labels': labels}