# Downloads the cutters from reddit

import praw
import requests
import os
import json
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()

reddit_client_id = os.getenv("REDDIT_CLIENT_ID")
reddit_client_secret = os.getenv("REDDIT_CLIENT_SECRET")
reddit_user_agent = os.getenv("REDDIT_USER_AGENT")

reddit = praw.Reddit(
    client_id=reddit_client_id,
    client_secret=reddit_client_secret,
    user_agent=reddit_user_agent
)

subreddit_name = "whatismycookiecutter"
flair = "Serious Answer First!"
download_limit = 500 

# Create a directory to save images
download_dir = "downloaded_images"
os.makedirs(download_dir, exist_ok=True)

# File to store processed post IDs
processed_file = "processed_posts.json"

# Load previously processed post IDs
def load_processed_posts():
    if os.path.exists(processed_file):
        with open(processed_file, "r") as f:
            return set(json.load(f))
    return set()

# Save processed post IDs
def save_processed_posts(processed_posts):
    with open(processed_file, "w") as f:
        json.dump(list(processed_posts), f)

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

def download_image(url, post_id, index=None):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Generate filename
        ext = os.path.splitext(urlparse(url).path)[1] or ".jpg"
        filename = f"{post_id}_{index or 0}{ext}" if index is not None else f"{post_id}{ext}"
        filepath = os.path.join(download_dir, filename)
        
        # Skip if file already exists
        if os.path.exists(filepath):
            print(f"Skipping {filepath} - already exists")
            return True
        
        # Save the image
        with open(filepath, "wb") as file:
            file.write(response.content)
        print(f"Downloaded image to {filepath}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Failed to download image from {url}: {e}")
        return False

# Load previously processed posts
processed_posts = load_processed_posts()
downloaded = 0

# Iterate through the posts with the specific flair
subreddit = reddit.subreddit(subreddit_name)
search_query = f'flair:"{flair}"'
# for post in subreddit.new():
for post in subreddit.search(search_query, limit=download_limit):
    if downloaded >= download_limit:
        break

    # Skip if post was already processed
    if post.id in processed_posts:
        print(f"Skipping already processed post: {post.title} (ID: {post.id})")
        continue

    print(f"\nProcessing post: {post.title}, URL: {post.url}")

    # Handle different types of posts
    try:
        # 1. Direct image links
        if post.url.endswith(VALID_EXTENSIONS):
            if download_image(post.url, post.id):
                processed_posts.add(post.id)
                downloaded += 1

        # 2. Reddit gallery posts
        elif hasattr(post, "is_gallery") and post.is_gallery:
            gallery_items = post.media_metadata
            all_downloaded = True
            for index, (item_id, item) in enumerate(gallery_items.items()):
                if item["status"] == "valid" and "s" in item:
                    image_url = item["s"]["u"].replace("&", "&")
                    if not download_image(image_url, post.id, index):
                        all_downloaded = False
                    else:
                        print(f"Downloaded gallery image {index+1}/{len(gallery_items)}")
            if all_downloaded:
                processed_posts.add(post.id)
                downloaded += 1

        # 3. Reddit-hosted single image posts (not direct URL)
        elif hasattr(post, "preview") and "images" in post.preview:
            image_url = post.preview["images"][0]["source"]["url"].replace("&", "&")
            if download_image(image_url, post.id):
                processed_posts.add(post.id)
                downloaded += 1

        else:
            print(f"Post type not supported for image download: {post.url}")

    except Exception as e:
        print(f"Error processing post {post.id}: {e}")
        continue

# Save the updated list of processed posts
save_processed_posts(processed_posts)
print(f"\nProcessed {downloaded} new posts with images. Total tracked posts: {len(processed_posts)}")