import subprocess

# Your specific Modal App IDs extracted directly from your URLs
app_ids = [
    "ap-6Hf0xia7F9yhnPnREVfA0w",
    "ap-ObdkiRwucenJV0rNjvjUsV",
    "ap-Q31rYxPbytYyQcc3hgZO8k",
    "ap-FH4DaC66jlm3nbXp9LJ4Zu",
    "ap-BWQN7zAV60cGHuX0OgJ6dH",
    "ap-TB7WoL5kg4ha0ZgQhPBkE6"
]

output_file = "all_training_logs.txt"

print(f"🚀 Starting log extraction for {len(app_ids)} apps...")

# Open the master text file
with open(output_file, "w", encoding="utf-8") as outfile:
    for app_id in app_ids:
        print(f"Fetching logs for {app_id}...")
        
        # Write a clean header so you know where one run ends and another begins
        outfile.write(f"\n\n{'='*60}\n")
        outfile.write(f"--- LOGS FOR RUN: {app_id} ---\n")
        outfile.write(f"{'='*60}\n\n")
        
        # Tell the Modal CLI to grab the logs
        try:
            # This runs 'modal app logs <app_id>' in the background
            result = subprocess.run(
                ["modal", "app", "logs", app_id], 
                capture_output=True, 
                text=True,
                check=True
            )
            outfile.write(result.stdout)
            print(f"✅ Success: {app_id}")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to fetch {app_id}. Error: {e}")
            outfile.write(f"[ERROR FETCHING LOGS FOR {app_id}]\n")

print(f"\n🎉 Done! All logs successfully stitched into {output_file}")