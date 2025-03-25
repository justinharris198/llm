import os
from dotenv import load_dotenv
from selenium import webdriver
from bs4 import BeautifulSoup
from openai import OpenAI
import time

load_dotenv(override=True)
api_key = os.getenv('OPENAI_API_KEY')

class ScrapedWebsiteAISummary:
    def __init__(self, url):
        self.url = url
        
        driver = webdriver.Chrome()  # or Firefox, Edge, etc.
        driver.get(url)
        time.sleep(10)
        
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        driver.quit()
        
        self.title = soup.title.string if soup.title else "No title found"
        for irrelevant in soup.body(["script", "style", "img", "input"]):
            irrelevant.decompose()
        self.text = soup.body.get_text(separator="\n", strip=True)
    
    def message(self, userPrompt, systemPrompt):
        self.message = [
            {"role": "system", "content": systemPrompt},
            {"role": "user", "content": userPrompt}
        ]
        
    def summarize(self):
        openai = OpenAI()
        
        self.response = openai.chat.completions.create(
            model = "gpt-4o-mini",
            messages = self.message
        )

url = 'https://www.wsj.com/'

data = ScrapedWebsiteAISummary(url)

userPrompt = userPrompt = f"You are looking at a website titled {data.title}"
userPrompt += "\nThe contents of this website is as follows; \
please provide a short summary of this website in markdown. \
If it includes news or announcements, then summarize these too.\n\n"
userPrompt += data.text

systemPrompt = "You are an assistant that analyzes the contents of a website \
and provides a short summary, ignoring text that might be navigation related. \
Respond in markdown."

data.message(userPrompt, systemPrompt)
data.summarize()