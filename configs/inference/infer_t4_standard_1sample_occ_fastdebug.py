_base_ = [
    "./infer_t4_standard_1sample_occ.py",
]

# Fast debug run for OCC sanity checks.
num_frames = 5
start_index = 0
end_index = 1

dataset = dict(
    num_frames=num_frames,
)

sampling_option = dict(
    num_frames=num_frames,
    num_steps=4,
    num_round=1,
)

outputs = "./outputs/t4_standard_candidate14_start33_occ_fastdebug"
save_dir = "./outputs/t4_standard_candidate14_start33_occ_fastdebug"
