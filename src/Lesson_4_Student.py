from dotenv import load_dotenv
_ = load_dotenv()

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI 
from langchain_community.tools.tavily_search import TavilySearchResults

tool = TavilySearchResults(max_results=2) 

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

# add persistence via checkpointer:
from langgraph.checkpoint.sqlite import SqliteSaver
memory = SqliteSaver.from_conn_string(":memory:")

class Agent:
    def __init__(self, model, tools, checkpointer, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges("llm", self.exists_action, {True: "action", False: END})
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile(checkpointer=checkpointer)
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def call_openai(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}

    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print("Back to the model!")
        return {'messages': results}

prompt = """You are a smart research assistant. Use the search engine to look up information. \
You are allowed to make multiple calls (either together or in sequence). \
Only look up information when you are sure of what you want. \
If you need to look up some information before asking a follow up question, you are allowed to do that!
"""
model = ChatOpenAI(model="gpt-4o") # no longer gpt-3.5-turbo
abot = Agent(model, [tool], system=prompt, checkpointer=memory)

messages = [HumanMessage(content="What is the weather in sf?")]

thread = {"configurable": {"thread_id": "1"}}
for event in abot.graph.stream({"messages": messages}, thread):
    for v in event.values():
        print(v['messages'])
# [AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_bmfLa92f6oAIKN9KvXtqbKDz', 'function': {'arguments': '{"query":"current weather in San Francisco"}', 'name': 'tavily_search_results_json'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 22, 'prompt_tokens': 151, 'total_tokens': 173, 'prompt_tokens_details': {'cached_tokens': 0, 'audio_tokens': 0}, 'completion_tokens_details': {'reasoning_tokens': 0, 'audio_tokens': 0, 'accepted_prediction_tokens': 0, 'rejected_prediction_tokens': 0}}, 'model_name': 'gpt-4o', 'system_fingerprint': 'fp_831e067d82', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-f59091ab-218a-4a30-b00e-4f8b10809383-0', tool_calls=[{'name': 'tavily_search_results_json', 'args': {'query': 'current weather in San Francisco'}, 'id': 'call_bmfLa92f6oAIKN9KvXtqbKDz'}])]

# Calling: {'name': 'tavily_search_results_json', 'args': {'query': 'current weather in San Francisco'}, 'id': 'call_bmfLa92f6oAIKN9KvXtqbKDz'}
# Back to the model!

# [ToolMessage(content='[{\'url\': \'https://www.weatherapi.com/\', \'content\': "{\'location\': {\'name\': \'San Francisco\', \'region\': \'California\', \'country\': \'United States of America\', \'lat\': 37.775, \'lon\': -122.4183, \'tz_id\': \'America/Los_Angeles\', \'localtime_epoch\': 1736065521, \'localtime\': \'2025-01-05 00:25\'}, \'current\': {\'last_updated_epoch\': 1736064900, \'last_updated\': \'2025-01-05 00:15\', \'temp_c\': 8.3, \'temp_f\': 46.9, \'is_day\': 0, \'condition\': {\'text\': \'Partly cloudy\', \'icon\': \'//cdn.weatherapi.com/weather/64x64/night/116.png\', \'code\': 1003}, \'wind_mph\': 3.1, \'wind_kph\': 5.0, \'wind_degree\': 44, \'wind_dir\': \'NE\', \'pressure_mb\': 1028.0, \'pressure_in\': 30.35, \'precip_mm\': 0.0, \'precip_in\': 0.0, \'humidity\': 86, \'cloud\': 25, \'feelslike_c\': 7.8, \'feelslike_f\': 46.1, \'windchill_c\': 10.3, \'windchill_f\': 50.6, \'heatindex_c\': 10.3, \'heatindex_f\': 50.6, \'dewpoint_c\': 8.9, \'dewpoint_f\': 48.0, \'vis_km\': 16.0, \'vis_miles\': 9.0, \'uv\': 0.0, \'gust_mph\': 5.2, \'gust_kph\': 8.4}}"}, {\'url\': \'https://www.easeweather.com/north-america/united-states/california/city-and-county-of-san-francisco/san-francisco/may\', \'content\': \'Weather in San Francisco in May 2025 - Detailed Forecast Weather in San Francisco for May 2025 Your guide to San Francisco weather in May - trends and predictions In general, the average temperature in San Francisco at the beginning of May is 17.1\\xa0°F. San Francisco in May average weather Access San Francisco weather forecasts with a simple click. Temperatures trend during May in San Francisco We recommend that you check the San Francisco forecast closer to your planned date for the most up-to-date weather information. Explore the daily rainfall trends and prepare for San Franciscos May weather\\xa0💧 Get accurate weather forecasts for San Francisco, located at latitude 37.775 and longitude -122.419.\'}]', name='tavily_search_results_json', tool_call_id='call_bmfLa92f6oAIKN9KvXtqbKDz')]

# [AIMessage(content='The current weather in San Francisco is partly cloudy with a temperature of 8.3°C (46.9°F). The wind is blowing from the northeast at 3.1 mph (5.0 kph), and the humidity is at 86%. The visibility is 16 kilometers (about 9 miles).', response_metadata={'token_usage': {'completion_tokens': 66, 'prompt_tokens': 775, 'total_tokens': 841, 'prompt_tokens_details': {'cached_tokens': 0, 'audio_tokens': 0}, 'completion_tokens_details': {'reasoning_tokens': 0, 'audio_tokens': 0, 'accepted_prediction_tokens': 0, 'rejected_prediction_tokens': 0}}, 'model_name': 'gpt-4o', 'system_fingerprint': 'fp_d28bcae782', 'finish_reason': 'stop', 'logprobs': None}, id='run-d959c3b6-1013-4016-8457-a269fdea17ba-0')]

# continuing the conversation, as we pass in thread_id 1 again -- so no need to mention "weather" again
(note that "messages" is `Annotated[list[AnyMessage], operator.add]`):
messages = [HumanMessage(content="What about in la?")]
thread = {"configurable": {"thread_id": "1"}}
for event in abot.graph.stream({"messages": messages}, thread):
    for v in event.values():
        # note that we now print v, not v["messages"]
        print(v)
# {'messages': [AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_ecSmspwOcDkqD0VfOEIkpVZs', 'function': {'arguments': '{"query":"current weather in Los Angeles"}', 'name': 'tavily_search_results_json'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 23, 'prompt_tokens': 852, 'total_tokens': 875, 'prompt_tokens_details': {'cached_tokens': 0, 'audio_tokens': 0}, 'completion_tokens_details': {'reasoning_tokens': 0, 'audio_tokens': 0, 'accepted_prediction_tokens': 0, 'rejected_prediction_tokens': 0}}, 'model_name': 'gpt-4o', 'system_fingerprint': 'fp_d28bcae782', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-7fb579b1-3fc4-4eca-808b-db947280116b-0', tool_calls=[{'name': 'tavily_search_results_json', 'args': {'query': 'current weather in Los Angeles'}, 'id': 'call_ecSmspwOcDkqD0VfOEIkpVZs'}])]}

# Calling: {'name': 'tavily_search_results_json', 'args': {'query': 'current weather in Los Angeles'}, 'id': 'call_ecSmspwOcDkqD0VfOEIkpVZs'}
# Back to the model!

# {'messages': [ToolMessage(content='[{\'url\': \'https://www.weatherapi.com/\', \'content\': "{\'location\': {\'name\': \'Los Angeles\', \'region\': \'California\', \'country\': \'United States of America\', \'lat\': 34.0522, \'lon\': -118.2428, \'tz_id\': \'America/Los_Angeles\', \'localtime_epoch\': 1736065632, \'localtime\': \'2025-01-05 00:27\'}, \'current\': {\'last_updated_epoch\': 1736064900, \'last_updated\': \'2025-01-05 00:15\', \'temp_c\': 11.7, \'temp_f\': 53.1, \'is_day\': 0, \'condition\': {\'text\': \'Clear\', \'icon\': \'//cdn.weatherapi.com/weather/64x64/night/113.png\', \'code\': 1000}, \'wind_mph\': 3.8, \'wind_kph\': 6.1, \'wind_degree\': 346, \'wind_dir\': \'NNW\', \'pressure_mb\': 1020.0, \'pressure_in\': 30.11, \'precip_mm\': 0.0, \'precip_in\': 0.0, \'humidity\': 74, \'cloud\': 0, \'feelslike_c\': 11.4, \'feelslike_f\': 52.5, \'windchill_c\': 11.1, \'windchill_f\': 52.0, \'heatindex_c\': 12.2, \'heatindex_f\': 53.9, \'dewpoint_c\': -3.8, \'dewpoint_f\': 25.1, \'vis_km\': 14.0, \'vis_miles\': 8.0, \'uv\': 0.0, \'gust_mph\': 7.7, \'gust_kph\': 12.4}}"}, {\'url\': \'https://www.weather25.com/north-america/usa/california/los-angeles?page=month&month=May\', \'content\': \'Full weather forecast for Los Angeles in May 2025. Check the temperatures, chance of rain and more in Los Angeles during May.\'}]', name='tavily_search_results_json', tool_call_id='call_ecSmspwOcDkqD0VfOEIkpVZs')]}
# {'messages': [AIMessage(content='The current weather in Los Angeles is clear with a temperature of 11.7°C (53.1°F). The wind is coming from the north-northwest at 3.8 mph (6.1 kph), and the humidity is 74%. The visibility is 14 kilometers (about 8 miles).', response_metadata={'token_usage': {'completion_tokens': 67, 'prompt_tokens': 1353, 'total_tokens': 1420, 'prompt_tokens_details': {'cached_tokens': 0, 'audio_tokens': 0}, 'completion_tokens_details': {'reasoning_tokens': 0, 'audio_tokens': 0, 'accepted_prediction_tokens': 0, 'rejected_prediction_tokens': 0}}, 'model_name': 'gpt-4o', 'system_fingerprint': 'fp_d28bcae782', 'finish_reason': 'stop', 'logprobs': None}, id='run-9400af1f-4591-4150-852f-bdc3f24aea6f-0')]}

messages = [HumanMessage(content="Which one is warmer?")]
thread = {"configurable": {"thread_id": "1"}}
for event in abot.graph.stream({"messages": messages}, thread):
    for v in event.values():
        print(v)
# {'messages': [AIMessage(content='Los Angeles is warmer than San Francisco at the moment. Los Angeles has a temperature of 11.7°C (53.1°F), while San Francisco has a temperature of 8.3°C (46.9°F).', response_metadata={'token_usage': {'completion_tokens': 47, 'prompt_tokens': 1431, 'total_tokens': 1478, 'prompt_tokens_details': {'cached_tokens': 1280, 'audio_tokens': 0}, 'completion_tokens_details': {'reasoning_tokens': 0, 'audio_tokens': 0, 'accepted_prediction_tokens': 0, 'rejected_prediction_tokens': 0}}, 'model_name': 'gpt-4o', 'system_fingerprint': 'fp_d28bcae782', 'finish_reason': 'stop', 'logprobs': None}, id='run-adfefafa-0786-4d66-8708-076408eee39d-0')]}

# changing to a new thread id:
messages = [HumanMessage(content="Which one is warmer?")]
thread = {"configurable": {"thread_id": "2"}}
for event in abot.graph.stream({"messages": messages}, thread):
    for v in event.values():
        print(v)
# {'messages': [AIMessage(content='Could you please specify the two options or locations you are comparing in terms of warmth?', response_metadata={'token_usage': {'completion_tokens': 18, 'prompt_tokens': 149, 'total_tokens': 167, 'prompt_tokens_details': {'cached_tokens': 0, 'audio_tokens': 0}, 'completion_tokens_details': {'reasoning_tokens': 0, 'audio_tokens': 0, 'accepted_prediction_tokens': 0, 'rejected_prediction_tokens': 0}}, 'model_name': 'gpt-4o', 'system_fingerprint': 'fp_831e067d82', 'finish_reason': 'stop', 'logprobs': None}, id='run-ae8a51ae-77da-4360-89d9-a4b26730a0f4-0')]}

# streaming tokens:

from langgraph.checkpoint.aiosqlite import AsyncSqliteSaver
memory = AsyncSqliteSaver.from_conn_string(":memory:")
abot = Agent(model, [tool], system=prompt, checkpointer=memory)

messages = [HumanMessage(content="What is the weather in SF?")]
thread = {"configurable": {"thread_id": "4"}}
async for event in abot.graph.astream_events({"messages": messages}, thread, version="v1"):
    kind = event["event"]
    if kind == "on_chat_model_stream":
        content = event["data"]["chunk"].content
        if content:
            # Empty content in the context of OpenAI means
            # that the model is asking for a tool to be invoked.
            # So we only print non-empty content
            # This is streaming in real-time
            print(content, end="|")
# /usr/local/lib/python3.11/site-packages/langchain_core/_api/beta_decorator.py:87: LangChainBetaWarning: This API is in beta and may change in the future.
#   warn_beta(
# Calling: {'name': 'tavily_search_results_json', 'args': {'query': 'current weather in San Francisco'}, 'id': 'call_HOnBA5512CLEsCfuGuDHwK4R'}
# Back to the model!

# The| current| weather| in| San| Francisco| is| partly| cloudy| with| a| temperature| of| |8|.|3|°C| (|46|.|9|°F|).| The| wind| is| blowing| from| the| northeast| at| |3|.|1| mph| (|5|.|0| k|ph|),| and| the| humidity| is| at| |86|%.|

# investigate event kinds:
thread = {"configurable": {"thread_id": "5"}}
async for event in abot.graph.astream_events({"messages": messages}, thread, version="v1"):
    kind = event["event"]
    print(f"Event {kind}")
    if kind == "on_chat_model_stream":
        content = event["data"]["chunk"].content
        if content:
            print(content, end="|")
# /usr/local/lib/python3.11/site-packages/langchain_core/_api/beta_decorator.py:87: LangChainBetaWarning: This API is in beta and may change in the future.
#   warn_beta(

# Event on_chain_start
# Event on_chain_start
# Event on_chain_end
# Event on_chain_start
# Event on_chat_model_start
# Event on_chat_model_stream
# Event on_chat_model_stream
# Event on_chat_model_stream
# Event on_chat_model_stream
# Event on_chat_model_stream
# Event on_chat_model_stream
# Event on_chat_model_stream
# Event on_chat_model_stream
# Event on_chat_model_stream
# Event on_chat_model_stream
# Event on_chat_model_stream
# Event on_chat_model_end
# Event on_chain_start
# Event on_chain_end
# Event on_chain_start
# Event on_chain_end
# Event on_chain_start
# Event on_chain_end
# Event on_chain_stream
# Event on_chain_end
# Event on_chain_stream
# Event on_chain_start
# Calling: {'name': 'tavily_search_results_json', 'args': {'query': 'current weather in San Francisco'}, 'id': 'call_HOnBA5512CLEsCfuGuDHwK4R'}
# Event on_tool_start
# Back to the model!
# Event on_tool_end
# Event on_chain_start
# Event on_chain_end
# Event on_chain_stream
# Event on_chain_end
# Event on_chain_stream
# Event on_chain_start
# Event on_chat_model_start
# Event on_chat_model_stream
# Event on_chat_model_stream
# The|Event on_chat_model_stream
#  current|Event on_chat_model_stream
#  weather|Event on_chat_model_stream
#  in|Event on_chat_model_stream
#  San|Event on_chat_model_stream
#  Francisco|Event on_chat_model_stream
#  is|Event on_chat_model_stream
#  partly|Event on_chat_model_stream
#  cloudy|Event on_chat_model_stream
#  with|Event on_chat_model_stream
#  a|Event on_chat_model_stream
#  temperature|Event on_chat_model_stream
#  of|Event on_chat_model_stream
#  |Event on_chat_model_stream
# 8|Event on_chat_model_stream
# .|Event on_chat_model_stream
# 3|Event on_chat_model_stream
# °C|Event on_chat_model_stream
#  (|Event on_chat_model_stream
# 46|Event on_chat_model_stream
# .|Event on_chat_model_stream
# 9|Event on_chat_model_stream
# °F|Event on_chat_model_stream
# ).|Event on_chat_model_stream
#  The|Event on_chat_model_stream
#  wind|Event on_chat_model_stream
#  is|Event on_chat_model_stream
#  blowing|Event on_chat_model_stream
#  from|Event on_chat_model_stream
#  the|Event on_chat_model_stream
#  northeast|Event on_chat_model_stream
#  at|Event on_chat_model_stream
#  |Event on_chat_model_stream
# 3|Event on_chat_model_stream
# .|Event on_chat_model_stream
# 1|Event on_chat_model_stream
#  mph|Event on_chat_model_stream
#  (|Event on_chat_model_stream
# 5|Event on_chat_model_stream
# .|Event on_chat_model_stream
# 0|Event on_chat_model_stream
#  k|Event on_chat_model_stream
# ph|Event on_chat_model_stream
# ),|Event on_chat_model_stream
#  and|Event on_chat_model_stream
#  the|Event on_chat_model_stream
#  humidity|Event on_chat_model_stream
#  is|Event on_chat_model_stream
#  at|Event on_chat_model_stream
#  |Event on_chat_model_stream
# 86|Event on_chat_model_stream
# %.|Event on_chat_model_stream
#  The|Event on_chat_model_stream
#  atmospheric|Event on_chat_model_stream
#  pressure|Event on_chat_model_stream
#  is|Event on_chat_model_stream
#  |Event on_chat_model_stream
# 102|Event on_chat_model_stream
# 8|Event on_chat_model_stream
# .|Event on_chat_model_stream
# 0|Event on_chat_model_stream
#  mb|Event on_chat_model_stream
# .|Event on_chat_model_stream
#  Visibility|Event on_chat_model_stream
#  is|Event on_chat_model_stream
#  good|Event on_chat_model_stream
#  at|Event on_chat_model_stream
#  |Event on_chat_model_stream
# 16|Event on_chat_model_stream
# .|Event on_chat_model_stream
# 0|Event on_chat_model_stream
#  km|Event on_chat_model_stream
#  (|Event on_chat_model_stream
# 9|Event on_chat_model_stream
# .|Event on_chat_model_stream
# 0|Event on_chat_model_stream
#  miles|Event on_chat_model_stream
# ).|Event on_chat_model_stream
# Event on_chat_model_end
# Event on_chain_start
# Event on_chain_end
# Event on_chain_start
# Event on_chain_end
# Event on_chain_stream
# Event on_chain_end
# Event on_chain_stream
# Event on_chain_end