import os, glob, torch, modal, numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

app = modal.App("data-packer-sharded")
volume = modal.Volume.from_name("libritts-volume")

@app.function(
    image=modal.Image.debian_slim().pip_install("torch", "numpy", "tqdm"), 
    volumes={"/data": volume}, 
    timeout=3600,
    cpu=4.0
)
def pack_data_sharded():
    data_dir = "/data/melspectrograms"
    mel_files = glob.glob(f"{data_dir}/*/*.npy")
    
    # Map speakers securely
    unique_speakers = sorted(list(set(os.path.basename(os.path.dirname(p)) for p in mel_files)))
    speaker_to_id = {spk: i for i, spk in enumerate(unique_speakers)}

    # We will make 10 shards
    num_shards = 10
    chunk_size = len(mel_files) // num_shards + 1
    
    print(f"📦 Sharding {len(mel_files)} files into {num_shards} safe chunks...")

    def load_single_file(path):
        try:
            spk = os.path.basename(os.path.dirname(path))
            arr = np.load(path)
            # float16 keeps file sizes extremely small and fast
            tensor = torch.from_numpy(arr).to(torch.float16).T 
            return (tensor, speaker_to_id[spk])
        except: 
            return None

    for i in range(num_shards):
        shard_files = mel_files[i*chunk_size : (i+1)*chunk_size]
        if not shard_files: continue
        
        shard_data = []
        # Multi-thread the reading
        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = [executor.submit(load_single_file, p) for p in shard_files]
            for f in tqdm(as_completed(futures), total=len(futures), desc=f"Packing Shard {i+1}/{num_shards}"):
                res = f.result()
                if res: shard_data.append(res)

        # Save just this small chunk
        shard_path = f"/data/dataset_shard_{i}.pt"
        print(f"💾 Saving Shard {i+1} to {shard_path}...")
        
        # Save mapping ONLY in shard 0 to avoid redundancy
        save_dict = {"data": shard_data}
        if i == 0: save_dict["mapping"] = speaker_to_id
            
        torch.save(save_dict, shard_path)
        volume.commit() # This clears the buffer and sends a successful Heartbeat!

    print("✅ All 10 shards saved successfully. The data is ready.")

@app.local_entrypoint()
def main():
    pack_data_sharded.remote()

if __name__ == "__main__":
    main()
    