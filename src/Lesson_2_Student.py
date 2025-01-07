from dotenv import load_dotenv
_ = load_dotenv()

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
# wrapper around openAI API, as LangChain has a standard interface for all LLMs it supports:
from langchain_openai import ChatOpenAI 
from langchain_community.tools.tavily_search import TavilySearchResults

tool = TavilySearchResults(max_results=2) #increased number of results
# "tavily_search_results_json" is the tool's name that the LLM will use:
print(type(tool))
# <class 'langchain_community.tools.tavily_search.tool.TavilySearchResults'>
print(tool.name) 
# tavily_search_results_json

class AgentState(TypedDict):
# why is messages not treated as an attribute of Agent (see below)? / 
# why are the Agent attributes (like tools, model) not in AgentState?
# Because messages is the only variable being mutated during graph traversal, i.e. during agent use!?
    messages: Annotated[list[AnyMessage], operator.add]
# see https://docs.python.org/3/library/typing.html

class Agent:

    def __init__(self, model, tools, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {True: "action", False: END}
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        # turn graph into a LangChain runnable (with a standard interface for invoking this graph):
        self.graph = graph.compile() 
        self.tools = {t.name: t for t in tools}
        # let the model know about the tools:
        self.model = model.bind_tools(tools) 

    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        # I guess AnyMessage has an attribute tool_calls:
        return len(result.tool_calls) > 0

    def call_openai(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        # due to operator.add, messages gets extended:
        return {'messages': [message]} 

    def take_action(self, state: AgentState):
        # parallel tool calls allowed (see also "together" in sys msg):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            # check for bad tool name from LLM:
            if not t['name'] in self.tools:      
                print("\n ....bad tool name....")
                # instruct LLM to retry if bad:
                result = "bad tool name, retry"  
            else:
                result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print("Back to the model!")
        # extends `messages` with up to len(tool_calls) messages:
        return {'messages': results} 

prompt = """You are a smart research assistant. Use the search engine to look up information. \
You are allowed to make multiple calls (either together or in sequence). \
Only look up information when you are sure of what you want. \
If you need to look up some information before asking a follow up question, you are allowed to do that!
"""

#reduce inference cost:
model = ChatOpenAI(model="gpt-3.5-turbo")  
abot = Agent(model, [tool], system=prompt)

from IPython.display import Image
Image(abot.graph.get_graph().draw_png())

# Prepare the state that the agent expects to work with:
messages = [HumanMessage(content="What is the weather in sf?")]
# Call agent by invoking graph:
result = abot.graph.invoke({"messages": messages})
# Calling: {'name': 'tavily_search_results_json', 'args': {'query': 'weather in San Francisco'}, 'id': 'call_PvPN1v7bHUxOdyn4J2xJhYOX'}
# Back to the model!
print(result)
# {'messages': [HumanMessage(content='What is the weather in sf?'),
#   AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_PvPN1v7bHUxOdyn4J2xJhYOX', 'function': {'arguments': '{"query":"weather in San Francisco"}', 'name': 'tavily_search_results_json'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 21, 'prompt_tokens': 153, 'total_tokens': 174, 'prompt_tokens_details': {'cached_tokens': 0, 'audio_tokens': 0}, 'completion_tokens_details': {'reasoning_tokens': 0, 'audio_tokens': 0, 'accepted_prediction_tokens': 0, 'rejected_prediction_tokens': 0}}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-175a85f8-1e3c-4c43-a9f4-dfd8c7efb91a-0', tool_calls=[{'name': 'tavily_search_results_json', 'args': {'query': 'weather in San Francisco'}, 'id': 'call_PvPN1v7bHUxOdyn4J2xJhYOX'}]),
#   ToolMessage(content='[{\'url\': \'https://www.weatherapi.com/\', \'content\': "{\'location\': {\'name\': \'San Francisco\', \'region\': \'California\', \'country\': \'United States of America\', \'lat\': 37.775, \'lon\': -122.4183, \'tz_id\': \'America/Los_Angeles\', \'localtime_epoch\': 1735996467, \'localtime\': \'2025-01-04 05:14\'}, \'current\': {\'last_updated_epoch\': 1735995600, \'last_updated\': \'2025-01-04 05:00\', \'temp_c\': 8.3, \'temp_f\': 46.9, \'is_day\': 0, \'condition\': {\'text\': \'Partly cloudy\', \'icon\': \'//cdn.weatherapi.com/weather/64x64/night/116.png\', \'code\': 1003}, \'wind_mph\': 2.9, \'wind_kph\': 4.7, \'wind_degree\': 309, \'wind_dir\': \'NW\', \'pressure_mb\': 1024.0, \'pressure_in\': 30.24, \'precip_mm\': 0.0, \'precip_in\': 0.0, \'humidity\': 83, \'cloud\': 50, \'feelslike_c\': 7.9, \'feelslike_f\': 46.3, \'windchill_c\': 8.8, \'windchill_f\': 47.9, \'heatindex_c\': 10.0, \'heatindex_f\': 50.0, \'dewpoint_c\': 6.9, \'dewpoint_f\': 44.3, \'vis_km\': 16.0, \'vis_miles\': 9.0, \'uv\': 0.0, \'gust_mph\': 4.7, \'gust_kph\': 7.6}}"}, {\'url\': \'https://www.easeweather.com/north-america/united-states/california/city-and-county-of-san-francisco/san-francisco/april\', \'content\': \'Weather in San Francisco in April 2025 - Detailed Forecast Weather in San Francisco for April 2025 Your guide to San Francisco weather in April - trends and predictions In general, the average temperature in San Francisco at the beginning of April is 15.9\\xa0°F. San Francisco experiences moderate rainfall in April, with an average of 8 rainy days and a total precipitation of 20.6\\xa0mm. San Francisco in April average weather Temperatures trend during April in San Francisco In April, San Francisco can expect light to moderate rainfall, totaling 20.6\\xa0mm across approximately 8 days. Explore the daily rainfall trends and prepare for San Franciscos April weather\\xa0💧 Get accurate weather forecasts for San Francisco, located at latitude 37.775 and longitude -122.419.\'}, {\'url\': \'https://world-weather.info/forecast/usa/san_francisco/april-2025/\', \'content\': "Weather in San Francisco in April 2025 (California) - Detailed Weather Forecast for a Month Weather World Weather in San Francisco Weather in San Francisco in April 2025 San Francisco Weather Forecast for April 2025, is based on previous years\' statistical data. +61°+52° +61°+52° +59°+50° +59°+52° +63°+52° +61°+52° +61°+52° +59°+50° +61°+52° +61°+52° +61°+54° +61°+52° +61°+52° +59°+50° +61°+52° +61°+52° +61°+52° +63°+54° +63°+54° +63°+54° +64°+54° +63°+54° +63°+54° +63°+54° +63°+54° +64°+54° +64°+54° +64°+54° +63°+54° +63°+54° Extended weather forecast in San Francisco HourlyWeek10-Day14-Day30-DayYear Weather in large and nearby cities Weather in Washington, D.C.+30° Sacramento+52° Pleasanton+52° Redwood City+54° San Leandro+54° San Mateo+54° San Rafael+50° San Ramon+50° South San Francisco+54° Vallejo+52° Palo Alto+52° Pacifica+52° Berkeley+52° Castro Valley+54° Concord+54° Daly City+52° Tiburon+50° Newport+52° world\'s temperature today day day Temperature units"}, {\'url\': \'https://www.weather25.com/north-america/usa/california/san-francisco?page=month&month=April\', \'content\': \'San Francisco weather in April 2025 | Weather25.com San Francisco weather in April 2025 The wather in San Francisco in April can vary between cold and nice weather days. | San Francisco in April | Temperatures in San Francisco in April Weather in San Francisco in April - FAQ The average temperature in San Francisco in April is 9/18° C. On average, there are 2 rainy days in San Francisco during April. The weather in San Francisco in April is good. On average, there are 0 snowy days in San Francisco in April. More about the weather in San Francisco San Francisco 14 day weather Long range weather for San Francisco San Francisco weather in November San Francisco weather in December San Francisco Webcam Weather tomorrow Hotels in San Francisco\'}]', name='tavily_search_results_json', tool_call_id='call_PvPN1v7bHUxOdyn4J2xJhYOX'),
#   AIMessage(content='The current weather in San Francisco is partly cloudy with a temperature of 46.9°F (8.3°C). The wind is blowing at 4.7 kph from the northwest direction. The humidity is at 83%, and there is no precipitation at the moment.', response_metadata={'token_usage': {'completion_tokens': 58, 'prompt_tokens': 1384, 'total_tokens': 1442, 'prompt_tokens_details': {'cached_tokens': 0, 'audio_tokens': 0}, 'completion_tokens_details': {'reasoning_tokens': 0, 'audio_tokens': 0, 'accepted_prediction_tokens': 0, 'rejected_prediction_tokens': 0}}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-11e718ba-ac2b-4f57-a13e-2e4f77f94f41-0')]}
print(result['messages'][-1].content)
# 'The current weather in San Francisco is partly cloudy with a temperature of 46.9°F (8.3°C). The wind is blowing at 4.7 kph from the northwest direction. The humidity is at 83%, and there is no precipitation at the moment.'

# Example of parallel function/tool calls:
messages = [HumanMessage(content="What is the weather in SF and LA?")]
result = abot.graph.invoke({"messages": messages})
# Calling: {'name': 'tavily_search_results_json', 'args': {'query': 'weather in San Francisco'}, 'id': 'call_dLM0iioal8hP5Nypf2CiFh1S'}
# Calling: {'name': 'tavily_search_results_json', 'args': {'query': 'weather in Los Angeles'}, 'id': 'call_Y2hZpq5Cx37z9iaGEFBdhYfB'}
# Back to the model!
print(result['messages'][-1].content)
# 'The current weather in San Francisco is overcast with a temperature of 51.3°F (10.7°C) and a light breeze. The humidity is at 80% with a cloud cover of 83%.\n\nIn Los Angeles, the weather is misty with a temperature of 59.7°F (15.4°C) and light winds. The humidity is at 62% with clear skies.\n\nIf you need more detailed information or forecasts, feel free to ask!'

# Example of sequention function/tool calls:
# Note, the query was modified to produce more consistent results. 
# Results may vary per run and over time as search information and models change:
query = "Who won the super bowl in 2024? In what state is the winning team headquarters located? \
What is the GDP of that state? Answer each question." 
messages = [HumanMessage(content=query)]

model = ChatOpenAI(model="gpt-4o")  # requires more advanced model
abot = Agent(model, [tool], system=prompt)
# executes sequentially (i.e. "Back to the model!" after each "Calling: ...")
# because 2nd query needs result from 1st query):
result = abot.graph.invoke({"messages": messages})
# Calling: {'name': 'tavily_search_results_json', 'args': {'query': '2024 Super Bowl winner'}, 'id': 'call_kXCyYHUKnmBbp6PFT6Ig2X4H'}
# Back to the model!
# Calling: {'name': 'tavily_search_results_json', 'args': {'query': 'Kansas City Chiefs headquarters location'}, 'id': 'call_CZfKWKxlGAeUyBhGfKpyugmJ'}
# Calling: {'name': 'tavily_search_results_json', 'args': {'query': 'Missouri GDP 2023'}, 'id': 'call_IZpS3sgdzQYAjtVi70wEKsgK'}
# Back to the model!
print(result['messages'][-1].content)
# 1. **Who won the Super Bowl in 2024?**
#    - The Kansas City Chiefs won the Super Bowl in 2024.
# 2. **In what state is the winning team's headquarters located?**
#    - The Kansas City Chiefs are headquartered in Kansas City, Missouri.
# 3. **What is the GDP of that state?**
#    - In 2023, the Gross Domestic Product (GDP) of Missouri was approximately $430 billion.
