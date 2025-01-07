from dotenv import load_dotenv
_ = load_dotenv()

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ChatMessage

memory = SqliteSaver.from_conn_string(":memory:")

class AgentState(TypedDict):
    task: str
    plan: str
    draft: str
    critique: str
    content: List[str]
    revision_number: int
    max_revisions: int

from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

PLAN_PROMPT = """You are an expert writer tasked with writing a high level outline of an essay. \
Write such an outline for the user provided topic. Give an outline of the essay along with any \ relevant notes or instructions for the sections."""

WRITER_PROMPT = """You are an essay assistant tasked with writing excellent 5-paragraph essays.\
Generate the best essay possible for the user's request and the initial outline. \
If the user provides critique, respond with a revised version of your previous attempts. \
Utilize all the information below as needed: 

------

{content}"""

REFLECTION_PROMPT = """You are a teacher grading an essay submission. \
Generate critique and recommendations for the user's submission. \
Provide detailed recommendations, including requests for length, depth, style, etc."""

RESEARCH_PLAN_PROMPT = """You are a researcher charged with providing information that can \
be used when writing the following essay. Generate a list of search queries that will gather \
any relevant information. Only generate 3 queries max."""

RESEARCH_CRITIQUE_PROMPT = """You are a researcher charged with providing information that can \
be used when making any requested revisions (as outlined below). \
Generate a list of search queries that will gather any relevant information. Only generate 3 queries max."""

# For generating the list of queries to pass to Tavily,
# we are going to use function calling to ensure we get back a list of strings from the LLM:
from langchain_core.pydantic_v1 import BaseModel
class Queries(BaseModel):
    queries: List[str]

# we are using the Tavily client instead of the tool because we are working with it
# in a slightly unconventional way:
from tavily import TavilyClient
import os
tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

def plan_node(state: AgentState):
    messages = [
        SystemMessage(content=PLAN_PROMPT), 
        HumanMessage(content=state['task'])
    ]
    response = model.invoke(messages)
    return {"plan": response.content}

def research_plan_node(state: AgentState):
    queries = model.with_structured_output(Queries).invoke([
        SystemMessage(content=RESEARCH_PLAN_PROMPT),
        HumanMessage(content=state['task'])
    ])
    content = state['content'] or []
    for q in queries.queries:
        response = tavily.search(query=q, max_results=2)
        for r in response['results']:
            content.append(r['content'])
    return {"content": content}

def generation_node(state: AgentState):
    content = "\n\n".join(state['content'] or [])
    user_message = HumanMessage(
        content=f"{state['task']}\n\nHere is my plan:\n\n{state['plan']}")
    messages = [
        SystemMessage(
            content=WRITER_PROMPT.format(content=content)
        ),
        user_message
        ]
    response = model.invoke(messages)
    return {
        "draft": response.content, 
        "revision_number": state.get("revision_number", 1) + 1
    }

def reflection_node(state: AgentState):
    messages = [
        SystemMessage(content=REFLECTION_PROMPT), 
        HumanMessage(content=state['draft'])
    ]
    response = model.invoke(messages)
    return {"critique": response.content}

def research_critique_node(state: AgentState):
    queries = model.with_structured_output(Queries).invoke([
        SystemMessage(content=RESEARCH_CRITIQUE_PROMPT),
        HumanMessage(content=state['critique'])
    ])
    content = state['content'] or []
    for q in queries.queries:
        response = tavily.search(query=q, max_results=2)
        for r in response['results']:
            content.append(r['content'])
    return {"content": content}

def should_continue(state):
    if state["revision_number"] > state["max_revisions"]:
        return END
    return "reflect"

builder = StateGraph(AgentState)
builder.add_node("planner", plan_node)
builder.add_node("generate", generation_node)
builder.add_node("reflect", reflection_node)
builder.add_node("research_plan", research_plan_node)
builder.add_node("research_critique", research_critique_node)
builder.set_entry_point("planner")
builder.add_conditional_edges(
    "generate", 
    should_continue, 
    {END: END, "reflect": "reflect"}
)
builder.add_edge("planner", "research_plan")
builder.add_edge("research_plan", "generate")
builder.add_edge("reflect", "research_critique")
builder.add_edge("research_critique", "generate")
graph = builder.compile(checkpointer=memory)

from IPython.display import Image
Image(graph.get_graph().draw_png())

thread = {"configurable": {"thread_id": "1"}}
for s in graph.stream({
    'task': "what is the difference between langchain and langsmith",
    "max_revisions": 2,
    "revision_number": 1,
}, thread):
    print(s)
# {'planner': {'plan': 'I. Introduction\n    A. Brief overview of Langchain and Langsmith\n    B. Thesis statement: Exploring the differences between Langchain and Langsmith\n\nII. Langchain\n    A. Definition and explanation\n    B. Key features and characteristics\n    C. Use cases and applications\n    D. Advantages and disadvantages\n\nIII. Langsmith\n    A. Definition and explanation\n    B. Key features and characteristics\n    C. Use cases and applications\n    D. Advantages and disadvantages\n\nIV. Comparison between Langchain and Langsmith\n    A. Technology stack\n    B. Scalability\n    C. Security\n    D. Interoperability\n    E. Performance\n\nV. Conclusion\n    A. Recap of main differences between Langchain and Langsmith\n    B. Implications for the future of blockchain technology\n    C. Final thoughts and recommendations\n\nNotes:\n- Ensure to provide clear definitions and examples for both Langchain and Langsmith.\n- Use specific examples and case studies to illustrate the differences between the two technologies.\n- Consider the latest developments and trends in blockchain technology to provide a comprehensive analysis.'}}
# {'research_plan': {'content': ['If you’re responsible for ensuring your AI models work in production, or you need to frequently debug and monitor your pipelines, Langsmith is your go-to tool. In short, while Langchain excels at managing and scaling model workflows, Langsmith is designed for those times when you need deep visibility and control over large, complex AI systems in production. But if you’re managing a complex AI pipeline with multiple models that need debugging and orchestrating, Langsmith’s capabilities become essential. If you’re debugging complex AI models or managing large-scale workflows with multiple moving parts, Langsmith’s advanced debugging and orchestration features will be indispensable. Additionally, if you’re working on cross-platform model deployments — say, running models on-prem and in the cloud simultaneously — Langsmith offers better orchestration and monitoring tools to handle the complexity.', "In this blog, we'll delve into the differences between LangChain and LangSmith, their pros and cons, and when to use each one. LangChain. LangChain is an open-source Python package that provides a framework for building and deploying LLM applications. It allows developers to create prototypes quickly and easily, making it an ideal choice for", 'LangChain and LangSmith are two complementary tools that cater to different stages and requirements of LLM development. LangChain is ideal for early-stage prototyping and small-scale applications, while LangSmith is better suited for large-scale, production-ready applications that require advanced debugging, testing, and monitoring capabilities', 'If you’re responsible for ensuring your AI models work in production, or you need to frequently debug and monitor your pipelines, Langsmith is your go-to tool. In short, while Langchain excels at managing and scaling model workflows, Langsmith is designed for those times when you need deep visibility and control over large, complex AI systems in production. But if you’re managing a complex AI pipeline with multiple models that need debugging and orchestrating, Langsmith’s capabilities become essential. If you’re debugging complex AI models or managing large-scale workflows with multiple moving parts, Langsmith’s advanced debugging and orchestration features will be indispensable. Additionally, if you’re working on cross-platform model deployments — say, running models on-prem and in the cloud simultaneously — Langsmith offers better orchestration and monitoring tools to handle the complexity.', 'LangChain and LangSmith are two complementary tools that cater to different stages and requirements of LLM development. LangChain is ideal for early-stage prototyping and small-scale applications, while LangSmith is better suited for large-scale, production-ready applications that require advanced debugging, testing, and monitoring capabilities', 'Discover the key differences between LangGraph, LangChain, LangFlow, and LangSmith, and learn which framework is best suited for your language model applications — from workflow building to performance monitoring. While the other tools focus on building workflows, LangSmith is designed for monitoring and debugging language model applications. Use LangGraph if you prefer graph-based, visual workflows for building complex LLM tasks. Use LangChain if you need a robust, flexible solution for creating language model applications programmatically. Each of these tools — LangGraph, LangChain, LangFlow, and LangSmith — caters to different stages of developing and managing language model applications. LangGraph provides a visual, intuitive way to build complex workflows, while LangChain offers a robust, code-first solution for developers looking to create scalable applications.']}}
# {'generate': {'draft': "**Essay: Exploring the Differences Between LangChain and LangSmith**\n\nIn the realm of AI model development and deployment, LangChain and LangSmith stand out as two distinct tools with unique functionalities and applications. This essay aims to delve into the disparities between LangChain and LangSmith, shedding light on their defining characteristics, use cases, advantages, and disadvantages.\n\n**LangChain**\n\nLangChain is an open-source Python package tailored for building and deploying LLM applications swiftly and efficiently. It serves as an excellent choice for developers engaged in early-stage prototyping and small-scale applications. The framework of LangChain enables developers to create prototypes rapidly, making it an ideal tool for those looking to experiment and iterate quickly in the initial stages of development.\n\nOne of the key features of LangChain is its flexibility and ease of use, allowing developers to create language model applications programmatically. This flexibility empowers developers to customize and tailor their applications to suit specific requirements. However, LangChain may not be the optimal choice for large-scale, production-ready applications that demand advanced debugging, testing, and monitoring capabilities.\n\n**LangSmith**\n\nOn the other hand, LangSmith is designed for managing and scaling complex AI systems in production environments. It offers deep visibility and control over intricate AI pipelines, making it indispensable for scenarios where advanced debugging and orchestration features are required. LangSmith excels in handling large-scale workflows with multiple moving parts, ensuring smooth operations and efficient monitoring.\n\nMoreover, LangSmith is particularly beneficial for cross-platform model deployments, where models need to run simultaneously on-premises and in the cloud. Its superior orchestration and monitoring tools make it a preferred choice for scenarios involving diverse deployment environments and complex system architectures.\n\n**Comparison Between LangChain and LangSmith**\n\nWhen comparing LangChain and LangSmith, several factors come into play. In terms of technology stack, LangChain focuses on providing a code-first solution for developers, while LangSmith emphasizes monitoring and debugging capabilities. Scalability-wise, LangSmith is better suited for large-scale applications, whereas LangChain is more tailored for smaller projects.\n\nSecurity and interoperability are also crucial considerations. LangSmith offers advanced security features to ensure the integrity of AI systems in production, while LangChain may require additional security measures for robust protection. In terms of interoperability, LangSmith's monitoring and orchestration tools enhance compatibility across different platforms, whereas LangChain may require additional integrations for seamless interoperability.\n\nIn conclusion, while LangChain and LangSmith are both valuable tools in the realm of AI model development, their distinct features and applications cater to different stages and requirements. Understanding the disparities between LangChain and LangSmith is essential for selecting the most suitable tool based on the specific needs of the project at hand. By leveraging the strengths of each tool, developers can optimize their AI model development and deployment processes effectively.", 'revision_number': 2}}

# {'reflect': {'critique': "**Critique:**\n\nThe essay provides a clear overview of the differences between LangChain and LangSmith in terms of their functionalities, applications, advantages, and disadvantages. The comparison between the two tools is well-structured and highlights key aspects that differentiate them. However, there are areas where the essay could be improved to provide a more comprehensive analysis.\n\n**Recommendations:**\n\n1. **Depth and Detail:** While the essay touches upon various aspects of LangChain and LangSmith, it would benefit from a deeper exploration of each tool's features. Providing specific examples or case studies to illustrate how each tool is used in real-world scenarios would enhance the reader's understanding.\n\n2. **Technical Analysis:** Consider including a more technical analysis of the tools, such as discussing the underlying algorithms, architecture, and performance metrics. This would appeal to a more technically inclined audience and provide a more in-depth comparison.\n\n3. **Use Cases:** Expand on the use cases of LangChain and LangSmith to showcase the practical applications of each tool. Discussing specific industries or projects where these tools have been successfully implemented would add value to the essay.\n\n4. **Pros and Cons:** While the essay briefly mentions the advantages and disadvantages of each tool, a more detailed analysis of the strengths and limitations would provide a balanced perspective. Consider delving deeper into the challenges that developers may face when using LangChain or LangSmith.\n\n5. **Conclusion:** Strengthen the conclusion by summarizing the key points of differentiation between LangChain and LangSmith and reiterating the importance of selecting the right tool based on project requirements. Consider offering recommendations or insights on how developers can make informed decisions when choosing between the two tools.\n\n6. **Length and Structure:** Expand on each section to provide a more comprehensive analysis. Consider breaking down the comparison into sub-sections for a more organized presentation of information.\n\nBy incorporating these recommendations, the essay can offer a more detailed and insightful analysis of the disparities between LangChain and LangSmith, providing readers with a deeper understanding of these AI model development tools."}}
# {'research_critique': {'content': ['If you’re responsible for ensuring your AI models work in production, or you need to frequently debug and monitor your pipelines, Langsmith is your go-to tool. In short, while Langchain excels at managing and scaling model workflows, Langsmith is designed for those times when you need deep visibility and control over large, complex AI systems in production. But if you’re managing a complex AI pipeline with multiple models that need debugging and orchestrating, Langsmith’s capabilities become essential. If you’re debugging complex AI models or managing large-scale workflows with multiple moving parts, Langsmith’s advanced debugging and orchestration features will be indispensable. Additionally, if you’re working on cross-platform model deployments — say, running models on-prem and in the cloud simultaneously — Langsmith offers better orchestration and monitoring tools to handle the complexity.', "In this blog, we'll delve into the differences between LangChain and LangSmith, their pros and cons, and when to use each one. LangChain. LangChain is an open-source Python package that provides a framework for building and deploying LLM applications. It allows developers to create prototypes quickly and easily, making it an ideal choice for", 'LangChain and LangSmith are two complementary tools that cater to different stages and requirements of LLM development. LangChain is ideal for early-stage prototyping and small-scale applications, while LangSmith is better suited for large-scale, production-ready applications that require advanced debugging, testing, and monitoring capabilities', 'If you’re responsible for ensuring your AI models work in production, or you need to frequently debug and monitor your pipelines, Langsmith is your go-to tool. In short, while Langchain excels at managing and scaling model workflows, Langsmith is designed for those times when you need deep visibility and control over large, complex AI systems in production. But if you’re managing a complex AI pipeline with multiple models that need debugging and orchestrating, Langsmith’s capabilities become essential. If you’re debugging complex AI models or managing large-scale workflows with multiple moving parts, Langsmith’s advanced debugging and orchestration features will be indispensable. Additionally, if you’re working on cross-platform model deployments — say, running models on-prem and in the cloud simultaneously — Langsmith offers better orchestration and monitoring tools to handle the complexity.', 'LangChain and LangSmith are two complementary tools that cater to different stages and requirements of LLM development. LangChain is ideal for early-stage prototyping and small-scale applications, while LangSmith is better suited for large-scale, production-ready applications that require advanced debugging, testing, and monitoring capabilities', 'Discover the key differences between LangGraph, LangChain, LangFlow, and LangSmith, and learn which framework is best suited for your language model applications — from workflow building to performance monitoring. While the other tools focus on building workflows, LangSmith is designed for monitoring and debugging language model applications. Use LangGraph if you prefer graph-based, visual workflows for building complex LLM tasks. Use LangChain if you need a robust, flexible solution for creating language model applications programmatically. Each of these tools — LangGraph, LangChain, LangFlow, and LangSmith — caters to different stages of developing and managing language model applications. LangGraph provides a visual, intuitive way to build complex workflows, while LangChain offers a robust, code-first solution for developers looking to create scalable applications.', 'Chains: LangChain’s core abstraction that allows developers to create sequences of operations, such as prompt generation, LLM invocation, and output processing. response = pipeline.run(text="LangChain is a framework for building LLM-powered applications.") response = llm_chain.run(input="What are LangChain\'s key features?") LangChain is a flexible tool for integrating large language models (LLMs) into AI applications, ideal for building complex workflows. The LangChain documentation provides guides, API references, and tutorials for everything from basic installation to advanced techniques like integrating external APIs. These tutorials are ideal for hands-on learners, guiding you through building real applications like chatbots or text generation systems. The ease of integration with LLMs, memory management, and support for diverse use cases make LangChain a strong choice for developers looking to build scalable and intelligent applications.', 'The following are some of the key features of LangChain:\nCustomizable prompts to suit your needs\nBuilding chain link components for advanced use cases\nCode customization for developing unique applications\nModel integration for data augmented generation and accessing high-quality language model application like text-davinci-003\nFlexible components to mix and match components for specific requirements\nContext manipulation to set and guide context for improved accuracy and user experience\nWith LangChain, you can create feature-rich applications that stand out from the crowd, thanks to its advanced customization options.\n Following are some of the key resources that you can use when working with LangChain:\nAI Libraries such as OpenAI and Hugging Face for AI models\nExternal sources such as Notion, Wikipedia, and Google Drive for targeted data\nLangChain documentation for guides on connecting and chaining components\nOfficial Documentation\nGitHub Repository\nPyPI Package Repository\nData augmentation to improve context-aware results through external data sources, indexing, and vector representations\nLastly, engaging with the LangChain community and dedicated support slack channel can be beneficial if you encounter challenges or want to learn from others’ experiences. Day(s)\n:\nHour(s)\n:\nMinute(s)\n:\nSecond(s)\nBlog\nDay(s)\n:\nHour(s)\n:\nMinute(s)\n:\nSecond(s)\n So, if you’re looking to stay ahead of the curve in the world of NLP, be sure to check out LangChain and see what it can do for you!\nRelated Posts\nText to Code Generator: Create Code in any Language\nAI\nHave you ever wondered if there was an easier way to write code? Generate Accurate Code in Seconds\nAI, Data Mentor\nWelcome to the exhilarating world of AI Code Generators, where imagination meets execution at lightning...\nJava Code Generator: How to Generate Java Code Quickly\nAI, Data Mentor, Java\nIntegrating artificial intelligence (AI) and natural language processing (NLP) has revolutionized how...\nC++ Code Generator:', 'How to Use Langsmith for Monitoring Langchain Applications | by Gary Svenson | Sep, 2024 | Medium How to Use Langsmith for Monitoring Langchain Applications How to Use Langsmith for Monitoring Langchain Applications Once you have gathered enough performance data, it is time to optimize your Langchain application based on the insights produced by Langsmith. API Usage Tracking: Langsmith can monitor API calls made by your Langchain applications, allowing you to grasp which APIs are most frequently accessed and to evaluate their performance. In summary, leveraging Langsmith for monitoring Langchain applications encompasses an in-depth understanding of both tools, proper installation, utilization of logging and error tracking features, systematic analysis, and optimization of workflows based on insights.', 'Like last year’s results, OpenAI reigns as the most used LLM provider among LangSmith users —\xa0used more than 6x as much as Ollama, the next-most popular provider (counted by LangSmith organization usage). While langchain (our open source framework) is central to many folks’ LLM app development journeys, 15.7% of LangSmith traces this year come from non-langchain frameworks. While it’s no easy feat to keep the quality of your LLM app high, we see organizations using LangSmith’s evaluation capabilities to automate testing and generate user feedback loops to create more robust, reliable applications. In 2024, developers leaned into complexity with multi-step agents, sharpened efficiency by doing more with fewer LLM calls, and added quality checks to their apps using methods of feedback and evaluation.', "However, the journey from prototype to production can be a challenging one. Enter LangSmith and LangChain, two platforms designed to streamline the development of LLM-powered applications. In this blog post, we'll explore some of the unique use-cases that can be created using these innovative tools. 1. Intelligent Agents for Data Analysis", "Thousands of companies build AI apps better with LangChain products. ### Rakuten Group builds with LangChain and LangSmith to deliver premium products for its business clients ### How Dun & Bradstreet's ChatD&B™ uses LangChain and LangSmith to deliver trusted, data-driven AI insights Not only did we deliver a better product by iterating with LangSmith, but we’re shipping new AI features to our users in a fraction of the time it would have taken without it.” “Working with LangChain and LangSmith on the Elastic AI Assistant had a significant positive impact on the overall pace and quality of the development and shipping experience. We couldn’t have achieved \xa0the product experience delivered to our customers without LangChain, and we couldn’t have done it at the same pace without LangSmith.”"]}}



# Essay Writer Interface

# Essay writer interface with gradio (will not work locally as dependencies are unavailable):
# needs interrupt_after = ['planner', 'generate', 'reflect', 'research_plan', 'research_critique']
# code is in helper.py

import warnings
warnings.filterwarnings("ignore")
from helper import ewriter, writer_gui

MultiAgent = ewriter()
app = writer_gui(MultiAgent.graph)
app.launch()