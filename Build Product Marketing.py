import os
from dotenv import load_dotenv
from selenium import webdriver
from bs4 import BeautifulSoup
import ollama
from openai import OpenAI
import json
from IPython.display import Markdown, display, update_display

load_dotenv(override=True)

class Llm:
    def __init__(self):
        os.getenv('OPENAI_API_KEY')
        self.openai = OpenAI()
        
    def setMessage(self, prompts):
        self.message = [
            {"role": "system", "content": prompts['system']},
            {"role": "user", "content": prompts['user']}
        ]
        
    def getResponseOpenAI(self, model):
        self.openAiResponse = self.openai.chat.completions.create(
            model = model,
            messages = self.message
        )
        
    def getResponseOpenAIStreamToMarkdown(self, model):
        stream = self.openai.chat.completions.create(
            model = model,
            messages = self.message,
            stream = True
        )
         
        response = ''
        display_handle = display(Markdown(''), display_id=True)
        
        for chunk in stream:
            response += chunk.choices[0].delta.content or ''
            response = response.replace("```","").replace("markdown", "")
            update_display(Markdown(response), display_id=display_handle.display_id)

    def getResponseOpenAIJson(self, model):
        openAiResponse = self.openai.chat.completions.create(
            model = model,
            messages = self.message,
            response_format={"type": "json_object"}
        )
        
        openAiResponseJson = openAiResponse.choices[0].message.content
        self.openAiResponseJson = json.loads(openAiResponseJson)
        
    def getResponseOllama(self, model):
        self.ollamaResponse = ollama.chat(model=model, messages=self.message)

class ScrapeWebData:
    def __init__(self, url):
        self.url = url
        
        driver = webdriver.Chrome()  # or Firefox, Edge, etc.
        driver.get(url)
        
        html = driver.page_source
        self.body = html
        soup = BeautifulSoup(self.body, 'html.parser')
        driver.quit()
        
        self.title = soup.title.string if soup.title else "No title found"
        
        if soup.body:
            for irrelevant in soup.body(["script", "style", "img", "input"]):
                irrelevant.decompose()
            self.text = soup.body.get_text(separator="\n", strip=True)
        else:
            self.text = ""
        
        links = [link.get('href') for link in soup.find_all('a')]
        self.links = [link for link in links if link]
        
    def getContents(self):
        return f"Webpage Title:\n{self.title}\nWebpage Contents:\n{self.text}\n\n"

def buildLinkPrompts(data):
    systemPrompt = "You are provided with a list of links found on a webpage. \
    You are able to decide which of the links would be most relevant to include in a brochure about the company, \
    such as links to an About page, or a Company page, or Careers/Jobs pages.\n"
    systemPrompt += "You should respond in JSON as in this example:"
    systemPrompt += """
    {
        "links": [
            {"type": "about page", "url": "https://full.url/goes/here/about"},
            {"type": "careers page": "url": "https://another.full.url/careers"}
        ]
    }
    """

    userPrompt = f"Here is the list of links on the website of {data.url} - "
    userPrompt += "please decide which of these are relevant web links for a brochure about the company, respond with the full https URL in JSON format. \
    Do not include Terms of Service, Privacy, email links.\n"
    userPrompt += "Links (some might be relative links):\n"
    userPrompt += "\n".join(data.links)
    
    return {'system': systemPrompt, 'user': userPrompt}

def buildProductPrompts(details, name):
    systemPrompt = "You are an assistant that analyzes the contents of several relevant pages from a company website \
    and creates a short brochure about the company for prospective customers, investors and recruits. Respond in markdown.\
    Include details of company culture, customers and careers/jobs if you have the information."


    userPrompt = f"You are looking at a company called: {name}\n"
    userPrompt += "Here are the contents of its landing page and other relevant pages; use this information to build a short brochure of the company in markdown.\n"
    userPrompt += details
    userPrompt = userPrompt[:5000] # Truncate if more than 5,000 characters
    
    return {'system': systemPrompt, 'user': userPrompt}

def buildUserPromptWebsiteData(data, links):
    result = "Landing page:\n"
    result += data.getContents()

    for link in links["links"]:
        result += f"\n\n{link['type']}\n"
        result += ScrapeWebData(link["url"]).getContents()
    
    return result


llm = Llm()
data = ScrapeWebData('https://huggingface.co')

llm.setMessage(buildLinkPrompts(data))
llm.getResponseOpenAIJson('gpt-4o-mini')

userPromptData = buildUserPromptWebsiteData(data, llm.openAiResponseJson)
llm.setMessage(buildProductPrompts(userPromptData, 'Hugging Face'))
llm.getResponseOpenAI('gpt-4o-mini')