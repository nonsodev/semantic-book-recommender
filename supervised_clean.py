

import pandas as pd
from googlesearch import search
import time
import random

df = pd.read_csv("search_progress.csv")
df1 = df.drop("query_index", axis=1)

print("Initial DataFrame:")
print(df1.head())

df1.columns = ["title", "url"]

unfinished = df1[(df1.isnull().any(axis=1)) | ~((df1["url"].str.contains("amazon", na=False)) | (df1["url"].str.contains("google", na=False)))]

unfinished_list = unfinished["title"].tolist()
unfinished_urls = [None] * len(unfinished_list)



for idx,i in enumerate(unfinished_list):
    print()
    print(f"Processing title {idx + 1}/{len(unfinished_list)}: {i}")
    try:
        results1 = search(i, num_results=3, lang="en")
        results2 = search(i.replace("google", "amazon"), num_results=3, lang="en")
        url = list(results1) + list(results2)
        count = 0
        print("\n")
        print(f"Searching for: {i}")
        for j in url:
            count += 1
            print(count, j)
        index = int(input("Enter the index of the correct URL (1-3): ")) - 1
        unfinished_urls[idx] = url[index]
    except Exception as e:
        print(f"Error occurred while searching for {i}: {e}")
        unfinished_urls[idx] = None 
    time.sleep(random.randint(1,5))  # Sleep to avoid hitting the search API too quickly

unfinished["url"] = unfinished_urls
print("Updated DataFrame with URLs:")
print(unfinished.head())

df1.update(unfinished)
df1.to_csv("search_progress1.csv", index=False)
    
    