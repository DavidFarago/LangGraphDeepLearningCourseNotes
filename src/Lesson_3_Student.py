from dotenv import load_dotenv
import os
from tavily import TavilyClient

# load tavily API key:
_ = load_dotenv() 
client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

# run search to test it out
result = client.search("What is in Nvidia's new Blackwell GPU?",
                       include_answer=True)
print(result["answer"])
# "Nvidia's new Blackwell GPU architecture supports real-time generative AI on trillion-parameter LLMs, offering up to 25 times the performance and energy efficiency compared to its predecessor. The Blackwell B100 and B200 enterprise GPUs make use of a dual-GPU Blackwell chip, but it is uncertain whether Nvidia will incorporate chiplets in gaming GPUs for this upcoming generation."

# regular (non-tavily) search:
city = "San Francisco"
query = f"""
    what is the current weather in {city}?
    Should I travel there today?
    "weather.com"
"""
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
import re
ddg = DDGS()
def search(query, max_results=6):
    try:
        results = ddg.text(query, max_results=max_results)
        return [i["href"] for i in results]
    except Exception as e:
        print(f"returning previous results due to exception reaching ddg.")
        results = [ # cover case where DDG rate limits due to high deeplearning.ai volume
            "https://weather.com/weather/today/l/USCA0987:1:US",
            "https://weather.com/weather/hourbyhour/l/54f9d8baac32496f6b5497b4bf7a277c3e2e6cc5625de69680e6169e7e38e9a8",
        ]
        return results  
for i in search(query):
    print(i)
# returning previous results due to exception reaching ddg.
# https://weather.com/weather/today/l/USCA0987:1:US
# https://weather.com/weather/hourbyhour/l/54f9d8baac32496f6b5497b4bf7a277c3e2e6cc5625de69680e6169e7e38e9a8

def scrape_weather_info(url):
    """Scrape content from the given URL"""
    if not url:
        return "Weather information could not be found."
    
    # fetch data
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return "Failed to retrieve the webpage."

    # parse result
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup

# use DuckDuckGo to find websites and take the first result
url = search(query)[0]
# returning previous results due to exception reaching ddg.
soup = scrape_weather_info(url)
print(f"Website: {url}\n\n")
# Website: https://weather.com/weather/today/l/USCA0987:1:US
# limit long outputs:
print(str(soup.body)[:2000]) 
# <body><div class="appWrapper DaybreakLargeScreen LargeScreen lightTheme twcTheme DaybreakLargeScreen--appWrapper--ZkDop gradients--cloudyFoggyDay--lBhxD gradients--cloudyFoggyDay-top---jGZr" id="appWrapper"><div class="region-meta"><div class="removeIfEmpty" id="WxuHtmlHead-meta-"></div><div class="removeIfEmpty" id="WxuNewsroom-meta-bc9f40d5-d941-4fd8-bae2-2d8d63a38bb3"></div></div><div class="region-topAds regionTopAds DaybreakLargeScreen--regionTopAds--sDajQ"><div class="removeIfEmpty" id="WxuAd-topAds-53dce052-5465-4609-a555-c3a20ab64ab0"><div class="adWrapper BaseAd--adWrapper--ANZ1O BaseAd--card--cqv7t BaseAd--hide--hCG8L"><div class="adLabel BaseAd--adLabel--JGSp6">Advertisement</div><div class="ad_module BaseAd--ad_module--ajh9S subs-undefined BaseAd--placeholder--ofteC" id="WX_Hidden"></div></div></div><div class="removeIfEmpty" id="WxuAd-topAds-fe926b10-58bc-448a-ab09-47e692334250"><div class="adWrapper BaseAd--adWrapper--ANZ1O BaseAd--card--cqv7t BaseAd--hide--hCG8L"><div class="adLabel BaseAd--adLabel--JGSp6">Advertisement</div><div class="ad_module BaseAd--ad_module--ajh9S subs-undefined BaseAd--placeholder--ofteC" id="MW_Interstitial"></div></div></div></div><div class="region-header regionHeader gradients--cloudyFoggyDay-top---jGZr" id="regionHeader"><div class="removeIfEmpty" id="WxuHeaderLargeScreen-header-9944ec87-e4d4-4f18-b23e-ce4a3fd8a3ba"><header aria-label="Menu" class="MainMenuHeader--MainMenuHeader--RBoq7 HeaderLargeScreen--HeaderLargeScreen--HPtiq gradients--cloudyFoggyDay-top---jGZr" role="banner"><div class="MainMenuHeader--wrapper--TVg8M"><div class="MainMenuHeader--wrapperLeft--frN1-"><a class="MainMenuHeader--accessibilityLink--bQU4R Button--secondary--dT8G-" href="#MainContent" target="_self">Skip to Main Content</a><a class="MainMenuHeader--accessibilityLink--bQU4R Button--secondary--dT8G-" href="https://www.essentialaccessibility.com/the-weather-channel?utm_source=theweatherchannelhomepage&amp;utm_medium=iconlarge&amp;utm_term=eacha

# extract text for better result, but the result will still not be concise enough:
weather_data = []
for tag in soup.find_all(['h1', 'h2', 'h3', 'p']):
    text = tag.get_text(" ", strip=True)
    weather_data.append(text)
weather_data = "\n".join(weather_data)
weather_data = re.sub(r'\s+', ' ', weather_data)
print(f"Website: {url}\n\n")
# Website: https://weather.com/weather/today/l/USCA0987:1:US
print(weather_data)
# recents Weather Forecasts Radar & Maps News & Media Products Health & Wellness Account Lifestyle Privacy Specialty Forecasts San Francisco, CA High Surf Advisory Today's Forecast for San Francisco, CA Morning Afternoon Evening Overnight Weather Today in San Francisco, CA 7:25 am 5:04 pm Don't Miss Hourly Forecast Now 2 pm 3 pm 4 pm 5 pm Outside That's Not What Was Expected Daily Forecast Today Sun 05 Mon 06 Tue 07 Wed 08 Radar We Love Our Critters Wellness In Winter Home, Garage & Garden Icy Nightmare From Utah To Kansas Health News For You Happening Near San Francisco, CA Popular Nextdoor posts Rain Or Shine, It's Playtime Stay Safe Worse Than Long COVID Weather in your inbox Your local forecast, plus daily trivia, stunning photos and our meteorologists’ top picks. All in one place, every weekday morning. By signing up, you're opting in to receive the Morning Brief email newsletter. To manage your data, visit Data Rights . Terms of Use | Privacy Policy Air Quality Index Air quality is considered satisfactory, and air pollution poses little or no risk. Health & Activities Seasonal Allergies and Pollen Count Forecast No pollen detected in your area Cold & Flu Forecast Flu risk is high in your area We recognize our responsibility to use data and technology for good. We may use or share your data with our data vendors. Take control of your data. The Weather Channel is the world's most accurate forecaster according to ForecastWatch, Global and Regional Weather Forecast Accuracy Overview , 2017-2022, commissioned by The Weather Company. © The Weather Company, LLC 2025

# AGENTIC SEARCH

# thus run tavily search to get exactly the kind of structured data an agent wants:
result = client.search(query, max_results=1)
data = result["results"][0]["content"]
print(data)
# {'location': {'name': 'San Francisco', 'region': 'California', 'country': 'United States of America', 'lat': 37.775, 'lon': -122.4183, 'tz_id': 'America/Los_Angeles', 'localtime_epoch': 1736026501, 'localtime': '2025-01-04 13:35'}, 'current': {'last_updated_epoch': 1736026200, 'last_updated': '2025-01-04 13:30', 'temp_c': 12.8, 'temp_f': 55.0, 'is_day': 1, 'condition': {'text': 'Partly cloudy', 'icon': '//cdn.weatherapi.com/weather/64x64/day/116.png', 'code': 1003}, 'wind_mph': 4.5, 'wind_kph': 7.2, 'wind_degree': 358, 'wind_dir': 'N', 'pressure_mb': 1026.0, 'pressure_in': 30.29, 'precip_mm': 0.0, 'precip_in': 0.0, 'humidity': 77, 'cloud': 75, 'feelslike_c': 12.4, 'feelslike_f': 54.4, 'windchill_c': 10.2, 'windchill_f': 50.4, 'heatindex_c': 11.0, 'heatindex_f': 51.8, 'dewpoint_c': 7.6, 'dewpoint_f': 45.7, 'vis_km': 16.0, 'vis_miles': 9.0, 'uv': 1.8, 'gust_mph': 5.9, 'gust_kph': 9.5}}

# pretty print:
import json
from pygments import highlight, lexers, formatters
parsed_json = json.loads(data.replace("'", '"'))
formatted_json = json.dumps(parsed_json, indent=4)
colorful_json = highlight(formatted_json,
                          lexers.JsonLexer(),
                          formatters.TerminalFormatter())
print(colorful_json)
# {
#     "location": {
#         "name": "San Francisco",
#         "region": "California",
#         "country": "United States of America",
#         "lat": 37.775,
#         "lon": -122.4183,
#         "tz_id": "America/Los_Angeles",
#         "localtime_epoch": 1736026501,
#         "localtime": "2025-01-04 13:35"
#     },
#     "current": {
#         "last_updated_epoch": 1736026200,
#         "last_updated": "2025-01-04 13:30",
#         "temp_c": 12.8,
#         "temp_f": 55.0,
#         "is_day": 1,
#         "condition": {
#             "text": "Partly cloudy",
#             "icon": "//cdn.weatherapi.com/weather/64x64/day/116.png",
#             "code": 1003
#         },
#         "wind_mph": 4.5,
#         "wind_kph": 7.2,
#         "wind_degree": 358,
#         "wind_dir": "N",
#         "pressure_mb": 1026.0,
#         "pressure_in": 30.29,
#         "precip_mm": 0.0,
#         "precip_in": 0.0,
#         "humidity": 77,
#         "cloud": 75,
#         "feelslike_c": 12.4,
#         "feelslike_f": 54.4,
#         "windchill_c": 10.2,
#         "windchill_f": 50.4,
#         "heatindex_c": 11.0,
#         "heatindex_f": 51.8,
#         "dewpoint_c": 7.6,
#         "dewpoint_f": 45.7,
#         "vis_km": 16.0,
#         "vis_miles": 9.0,
#         "uv": 1.8,
#         "gust_mph": 5.9,
#         "gust_kph": 9.5
#     }
# }

