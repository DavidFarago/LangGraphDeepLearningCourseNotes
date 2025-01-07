# from dotenv import load_dotenv
_ = load_dotenv()

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langgraph.checkpoint.sqlite import SqliteSaver

# Build a small simple graph (2 nodes) if you want more insight into controlling state memory:
# Use as state: 
#   -lnode: last node 
#   -scratch: a scratchpad location 
#   -count : a counter that is incremented each step
class AgentState(TypedDict):
    lnode: str
    scratch: str
    count: Annotated[int, operator.add]

def node1(state: AgentState):
    print(f"node1, count:{state['count']}")
    return {"lnode": "node_1",
            "count": 1,
           }
def node2(state: AgentState):
    print(f"node2, count:{state['count']}")
    return {"lnode": "node_2",
            "count": 1,
           }

# The graph goes N1->N2->N1... but breaks after count reaches 3:
def should_continue(state):
    return state["count"] < 3

builder = StateGraph(AgentState)
builder.add_node("Node1", node1)
builder.add_node("Node2", node2)

builder.add_edge("Node1", "Node2")
builder.add_conditional_edges("Node2", 
                              should_continue, 
                              {True: "Node1", False: END})
builder.set_entry_point("Node1")

memory = SqliteSaver.from_conn_string(":memory:")
graph = builder.compile(checkpointer=memory)

# run the graph:
thread = {"configurable": {"thread_id": str(1)}}
print(graph.invoke({"count":0, "scratch":"hi"},thread))
# node1, count:0
# node2, count:1
# node1, count:2
# node2, count:3

# {'lnode': 'node_2', 'scratch': 'hi', 'count': 4}
print(graph.get_state(thread))
StateSnapshot(values={'lnode': 'node_2', 'scratch': 'hi', 'count': 4}, next=(), config={'configurable': {'thread_id': '1', 'thread_ts': '1efcc47f-df19-6813-8004-be30dd12514a'}}, metadata={'source': 'loop', 'step': 4, 'writes': {'Node2': {'count': 1, 'lnode': 'node_2'}}}, created_at='2025-01-06T16:05:04.698758+00:00', parent_config={'configurable': {'thread_id': '1', 'thread_ts': '1efcc47f-df15-6ca9-8003-a37be7190f50'}})

# View all the statesnapshots in memory. You can use the displayed count agentstate variable to help track what you see. Notice the most recent snapshots are returned by the iterator first. Also note that there is a handy step variable in the metadata that counts the number of steps in the graph execution. This is a bit detailed - but you can also notice that the parent_config is the config of the previous node. At initial startup, additional states are inserted into memory to create a parent. This is something to check when you branch or time travel below:
for state in graph.get_state_history(thread):
    print(state, "\n")
# StateSnapshot(values={'lnode': 'node_2', 'scratch': 'hi', 'count': 4}, next=(), config={'configurable': {'thread_id': '1', 'thread_ts': '1efcc47f-df19-6813-8004-be30dd12514a'}}, metadata={'source': 'loop', 'step': 4, 'writes': {'Node2': {'count': 1, 'lnode': 'node_2'}}}, created_at='2025-01-06T16:05:04.698758+00:00', parent_config={'configurable': {'thread_id': '1', 'thread_ts': '1efcc47f-df15-6ca9-8003-a37be7190f50'}}) 

# StateSnapshot(values={'lnode': 'node_1', 'scratch': 'hi', 'count': 3}, next=('Node2',), config={'configurable': {'thread_id': '1', 'thread_ts': '1efcc47f-df15-6ca9-8003-a37be7190f50'}}, metadata={'source': 'loop', 'step': 3, 'writes': {'Node1': {'count': 1, 'lnode': 'node_1'}}}, created_at='2025-01-06T16:05:04.697233+00:00', parent_config={'configurable': {'thread_id': '1', 'thread_ts': '1efcc47f-df12-632b-8002-f3869d6f8771'}}) 

# StateSnapshot(values={'lnode': 'node_2', 'scratch': 'hi', 'count': 2}, next=('Node1',), config={'configurable': {'thread_id': '1', 'thread_ts': '1efcc47f-df12-632b-8002-f3869d6f8771'}}, metadata={'source': 'loop', 'step': 2, 'writes': {'Node2': {'count': 1, 'lnode': 'node_2'}}}, created_at='2025-01-06T16:05:04.695766+00:00', parent_config={'configurable': {'thread_id': '1', 'thread_ts': '1efcc47f-df0b-6ead-8001-86f1636b1b4f'}}) 

# StateSnapshot(values={'lnode': 'node_1', 'scratch': 'hi', 'count': 1}, next=('Node2',), config={'configurable': {'thread_id': '1', 'thread_ts': '1efcc47f-df0b-6ead-8001-86f1636b1b4f'}}, metadata={'source': 'loop', 'step': 1, 'writes': {'Node1': {'count': 1, 'lnode': 'node_1'}}}, created_at='2025-01-06T16:05:04.693189+00:00', parent_config={'configurable': {'thread_id': '1', 'thread_ts': '1efcc47f-df06-6652-8000-b73e3dc95d28'}}) 

# StateSnapshot(values={'scratch': 'hi', 'count': 0}, next=('Node1',), config={'configurable': {'thread_id': '1', 'thread_ts': '1efcc47f-df06-6652-8000-b73e3dc95d28'}}, metadata={'source': 'loop', 'step': 0, 'writes': None}, created_at='2025-01-06T16:05:04.690932+00:00', parent_config={'configurable': {'thread_id': '1', 'thread_ts': '1efcc47f-df02-6a83-bfff-5cfe468bc45b'}}) 

# StateSnapshot(values={'count': 0}, next=('__start__',), config={'configurable': {'thread_id': '1', 'thread_ts': '1efcc47f-df02-6a83-bfff-5cfe468bc45b'}}, metadata={'source': 'input', 'step': -1, 'writes': {'count': 0, 'scratch': 'hi'}}, created_at='2025-01-06T16:05:04.689400+00:00', parent_config=None) 

# Store just the config into a list. Note the sequence of counts on the right: get_state_history returns the most recent snapshots first:
states = []
for state in graph.get_state_history(thread):
    states.append(state.config)
    print(state.config, state.values['count'])
# {'configurable': {'thread_id': '1', 'thread_ts': '1efcc47f-df19-6813-8004-be30dd12514a'}} 4
# {'configurable': {'thread_id': '1', 'thread_ts': '1efcc47f-df15-6ca9-8003-a37be7190f50'}} 3
# {'configurable': {'thread_id': '1', 'thread_ts': '1efcc47f-df12-632b-8002-f3869d6f8771'}} 2
# {'configurable': {'thread_id': '1', 'thread_ts': '1efcc47f-df0b-6ead-8001-86f1636b1b4f'}} 1
# {'configurable': {'thread_id': '1', 'thread_ts': '1efcc47f-df06-6652-8000-b73e3dc95d28'}} 0
# {'configurable': {'thread_id': '1', 'thread_ts': '1efcc47f-df02-6a83-bfff-5cfe468bc45b'}} 0
print(states[-3])
# {'configurable': {'thread_id': '1',
#   'thread_ts': '1efcc47f-df0b-6ead-8001-86f1636b1b4f'}}

# This is the state after Node1 completed for the first time. Note next is Node2and count is 1:
print(graph.get_state(states[-3]))
# StateSnapshot(values={'lnode': 'node_1', 'scratch': 'hi', 'count': 1}, next=('Node2',), config={'configurable': {'thread_id': '1', 'thread_ts': '1efcc47f-df0b-6ead-8001-86f1636b1b4f'}}, metadata={'source': 'loop', 'step': 1, 'writes': {'Node1': {'count': 1, 'lnode': 'node_1'}}}, created_at='2025-01-06T16:05:04.693189+00:00', parent_config={'configurable': {'thread_id': '1', 'thread_ts': '1efcc47f-df06-6652-8000-b73e3dc95d28'}})

# Go Back in Time:

# uses states[-3] as current_state; this continues to node2:
graph.invoke(None, states[-3])
# node2, count:1
# node1, count:2
# node2, count:3

# {'lnode': 'node_2', 'scratch': 'hi', 'count': 4}

# new states are now in state history. Notice the counts on the far right:
thread = {"configurable": {"thread_id": str(1)}}
for state in graph.get_state_history(thread):
    print(state.config, state.values['count'])
# {'configurable': {'thread_id': '1', 'thread_ts': '1efcc4a2-870c-6bc2-8004-ccbd21e9c0ec'}} 4
# {'configurable': {'thread_id': '1', 'thread_ts': '1efcc4a2-8708-6209-8003-1c1060c510b8'}} 3
# {'configurable': {'thread_id': '1', 'thread_ts': '1efcc4a2-8704-6e6d-8002-ac04f82f3159'}} 2
# {'configurable': {'thread_id': '1', 'thread_ts': '1efcc47f-df19-6813-8004-be30dd12514a'}} 4
# {'configurable': {'thread_id': '1', 'thread_ts': '1efcc47f-df15-6ca9-8003-a37be7190f50'}} 3
# {'configurable': {'thread_id': '1', 'thread_ts': '1efcc47f-df12-632b-8002-f3869d6f8771'}} 2
# {'configurable': {'thread_id': '1', 'thread_ts': '1efcc47f-df0b-6ead-8001-86f1636b1b4f'}} 1
# {'configurable': {'thread_id': '1', 'thread_ts': '1efcc47f-df06-6652-8000-b73e3dc95d28'}} 0
# {'configurable': {'thread_id': '1', 'thread_ts': '1efcc47f-df02-6a83-bfff-5cfe468bc45b'}} 0

# the details below show the node that start the new branch: the parent config is not the previous entry in the stack, but is the entry from state[-3]:
thread = {"configurable": {"thread_id": str(1)}}
for state in graph.get_state_history(thread):
    print(state,"\n")
# StateSnapshot(values={'lnode': 'node_2', 'scratch': 'hi', 'count': 4}, next=(), config={'configurable': {'thread_id': '1', 'thread_ts': '1efcc4a2-870c-6bc2-8004-ccbd21e9c0ec'}}, metadata={'source': 'loop', 'step': 4, 'writes': {'Node2': {'count': 1, 'lnode': 'node_2'}}}, created_at='2025-01-06T16:20:34.990167+00:00', parent_config={'configurable': {'thread_id': '1', 'thread_ts': '1efcc4a2-8708-6209-8003-1c1060c510b8'}}) 

# StateSnapshot(values={'lnode': 'node_1', 'scratch': 'hi', 'count': 3}, next=('Node2',), config={'configurable': {'thread_id': '1', 'thread_ts': '1efcc4a2-8708-6209-8003-1c1060c510b8'}}, metadata={'source': 'loop', 'step': 3, 'writes': {'Node1': {'count': 1, 'lnode': 'node_1'}}}, created_at='2025-01-06T16:20:34.988276+00:00', parent_config={'configurable': {'thread_id': '1', 'thread_ts': '1efcc4a2-8704-6e6d-8002-ac04f82f3159'}}) 

# StateSnapshot(values={'lnode': 'node_2', 'scratch': 'hi', 'count': 2}, next=('Node1',), config={'configurable': {'thread_id': '1', 'thread_ts': '1efcc4a2-8704-6e6d-8002-ac04f82f3159'}}, metadata={'source': 'loop', 'step': 2, 'writes': {'Node2': {'count': 1, 'lnode': 'node_2'}}}, created_at='2025-01-06T16:20:34.986948+00:00', parent_config={'configurable': {'thread_id': '1', 'thread_ts': '1efcc47f-df0b-6ead-8001-86f1636b1b4f'}}) 

# StateSnapshot(values={'lnode': 'node_2', 'scratch': 'hi', 'count': 4}, next=(), config={'configurable': {'thread_id': '1', 'thread_ts': '1efcc47f-df19-6813-8004-be30dd12514a'}}, metadata={'source': 'loop', 'step': 4, 'writes': {'Node2': {'count': 1, 'lnode': 'node_2'}}}, created_at='2025-01-06T16:05:04.698758+00:00', parent_config={'configurable': {'thread_id': '1', 'thread_ts': '1efcc47f-df15-6ca9-8003-a37be7190f50'}}) 

# StateSnapshot(values={'lnode': 'node_1', 'scratch': 'hi', 'count': 3}, next=('Node2',), config={'configurable': {'thread_id': '1', 'thread_ts': '1efcc47f-df15-6ca9-8003-a37be7190f50'}}, metadata={'source': 'loop', 'step': 3, 'writes': {'Node1': {'count': 1, 'lnode': 'node_1'}}}, created_at='2025-01-06T16:05:04.697233+00:00', parent_config={'configurable': {'thread_id': '1', 'thread_ts': '1efcc47f-df12-632b-8002-f3869d6f8771'}}) 

# StateSnapshot(values={'lnode': 'node_2', 'scratch': 'hi', 'count': 2}, next=('Node1',), config={'configurable': {'thread_id': '1', 'thread_ts': '1efcc47f-df12-632b-8002-f3869d6f8771'}}, metadata={'source': 'loop', 'step': 2, 'writes': {'Node2': {'count': 1, 'lnode': 'node_2'}}}, created_at='2025-01-06T16:05:04.695766+00:00', parent_config={'configurable': {'thread_id': '1', 'thread_ts': '1efcc47f-df0b-6ead-8001-86f1636b1b4f'}}) 

# StateSnapshot(values={'lnode': 'node_1', 'scratch': 'hi', 'count': 1}, next=('Node2',), config={'configurable': {'thread_id': '1', 'thread_ts': '1efcc47f-df0b-6ead-8001-86f1636b1b4f'}}, metadata={'source': 'loop', 'step': 1, 'writes': {'Node1': {'count': 1, 'lnode': 'node_1'}}}, created_at='2025-01-06T16:05:04.693189+00:00', parent_config={'configurable': {'thread_id': '1', 'thread_ts': '1efcc47f-df06-6652-8000-b73e3dc95d28'}}) 

# StateSnapshot(values={'scratch': 'hi', 'count': 0}, next=('Node1',), config={'configurable': {'thread_id': '1', 'thread_ts': '1efcc47f-df06-6652-8000-b73e3dc95d28'}}, metadata={'source': 'loop', 'step': 0, 'writes': None}, created_at='2025-01-06T16:05:04.690932+00:00', parent_config={'configurable': {'thread_id': '1', 'thread_ts': '1efcc47f-df02-6a83-bfff-5cfe468bc45b'}}) 

# StateSnapshot(values={'count': 0}, next=('__start__',), config={'configurable': {'thread_id': '1', 'thread_ts': '1efcc47f-df02-6a83-bfff-5cfe468bc45b'}}, metadata={'source': 'input', 'step': -1, 'writes': {'count': 0, 'scratch': 'hi'}}, created_at='2025-01-06T16:05:04.689400+00:00', parent_config=None) 

# Modify state:

# start fresh with thread_id 2
thread2 = {"configurable": {"thread_id": str(2)}}
graph.invoke({"count":0, "scratch":"hi"},thread2)
# node1, count:0
# node2, count:1
# node1, count:2
# node2, count:3

# {'lnode': 'node_2', 'scratch': 'hi', 'count': 4}

from IPython.display import Image
Image(graph.get_graph().draw_png())

states2 = []
for state in graph.get_state_history(thread2):
    states2.append(state.config)
    print(state.config, state.values['count'])
# {'configurable': {'thread_id': '2', 'thread_ts': '1efcc5ff-0eac-6c10-8004-8a83ce78de26'}} 4
# {'configurable': {'thread_id': '2', 'thread_ts': '1efcc5ff-0ea7-6f1e-8003-967364705237'}} 3
# {'configurable': {'thread_id': '2', 'thread_ts': '1efcc5ff-0ea4-6748-8002-3ee8b7f88f6f'}} 2
# {'configurable': {'thread_id': '2', 'thread_ts': '1efcc5ff-0ea0-6e88-8001-ec0e11465a39'}} 1
# {'configurable': {'thread_id': '2', 'thread_ts': '1efcc5ff-0e9c-6fa4-8000-749eeaf378c0'}} 0
# {'configurable': {'thread_id': '2', 'thread_ts': '1efcc5ff-0e99-6fce-bfff-54f64ddc5148'}} 0    

save_state = graph.get_state(states2[-3])
print(save_state)
# StateSnapshot(values={'lnode': 'node_1', 'scratch': 'hi', 'count': 1}, next=('Node2',), config={'configurable': {'thread_id': '2', 'thread_ts': '1efcc5ff-0ea0-6e88-8001-ec0e11465a39'}}, metadata={'source': 'loop', 'step': 1, 'writes': {'Node1': {'count': 1, 'lnode': 'node_1'}}}, created_at='2025-01-06T18:56:30.760499+00:00', parent_config={'configurable': {'thread_id': '2', 'thread_ts': '1efcc5ff-0e9c-6fa4-8000-749eeaf378c0'}})

save_state.values["count"] = -3
save_state.values["scratch"] = "hello"
print(save_state)
# StateSnapshot(values={'lnode': 'node_1', 'scratch': 'hello', 'count': -3}, next=('Node2',), config={'configurable': {'thread_id': '2', 'thread_ts': '1efcc5ff-0ea0-6e88-8001-ec0e11465a39'}}, metadata={'source': 'loop', 'step': 1, 'writes': {'Node1': {'count': 1, 'lnode': 'node_1'}}}, created_at='2025-01-06T18:56:30.760499+00:00', parent_config={'configurable': {'thread_id': '2', 'thread_ts': '1efcc5ff-0e9c-6fa4-8000-749eeaf378c0'}})

# update the state of the graph by creating a new entry at the top (latest entry) in memory:
print(graph.update_state(thread2,save_state.values))
# {'configurable': {'thread_id': '2',
#   'thread_ts': '1efcc617-1f54-6bf2-8005-15f572acfc33'}}

# the new entry on top has
# - the previous top as parent
# - count 1 because update -3 was added to the previous count value rather than replaceing it.

for i, state in enumerate(graph.get_state_history(thread2)):
    if i >= 3:
        break
    print(state, '\n')
# StateSnapshot(values={'lnode': 'node_1', 'scratch': 'hello', 'count': 1}, next=('Node1',), config={'configurable': {'thread_id': '2', 'thread_ts': '1efcc617-1f54-6bf2-8005-15f572acfc33'}}, metadata={'source': 'update', 'step': 5, 'writes': {'Node2': {'count': -3, 'lnode': 'node_1', 'scratch': 'hello'}}}, created_at='2025-01-06T19:07:16.756980+00:00', parent_config={'configurable': {'thread_id': '2', 'thread_ts': '1efcc5ff-0eac-6c10-8004-8a83ce78de26'}}) 

# StateSnapshot(values={'lnode': 'node_2', 'scratch': 'hi', 'count': 4}, next=(), config={'configurable': {'thread_id': '2', 'thread_ts': '1efcc5ff-0eac-6c10-8004-8a83ce78de26'}}, metadata={'source': 'loop', 'step': 4, 'writes': {'Node2': {'count': 1, 'lnode': 'node_2'}}}, created_at='2025-01-06T18:56:30.765356+00:00', parent_config={'configurable': {'thread_id': '2', 'thread_ts': '1efcc5ff-0ea7-6f1e-8003-967364705237'}}) 

# StateSnapshot(values={'lnode': 'node_1', 'scratch': 'hi', 'count': 3}, next=('Node2',), config={'configurable': {'thread_id': '2', 'thread_ts': '1efcc5ff-0ea7-6f1e-8003-967364705237'}}, metadata={'source': 'loop', 'step': 3, 'writes': {'Node1': {'count': 1, 'lnode': 'node_1'}}}, created_at='2025-01-06T18:56:30.763384+00:00', parent_config={'configurable': {'thread_id': '2', 'thread_ts': '1efcc5ff-0ea4-6748-8002-3ee8b7f88f6f'}}) 

# Try again with as_node:

# The metadata attribute "writes" defines what situation the writer should assume.
# When writing using update_state(), you want to define to the graph logic which node should be assumed as the writer. What this does is allow th graph logic to find the node on the graph. After writing the values, the next() value is computed by travesing the graph using the new state. In this case, the state we have was written by Node1. The graph can then compute the next state as being Node2:
print(graph.update_state(thread2, save_state.values, as_node="Node1"))
# {'configurable': {'thread_id': '2',
#   'thread_ts': '1efcc62a-effe-63cf-8006-ebe2d6f52a84'}}

for i, state in enumerate(graph.get_state_history(thread2)):
    if i >= 3:
        break
    print(state, '\n')
# StateSnapshot(values={'lnode': 'node_1', 'scratch': 'hello', 'count': -2}, next=('Node2',), config={'configurable': {'thread_id': '2', 'thread_ts': '1efcc62a-effe-63cf-8006-ebe2d6f52a84'}}, metadata={'source': 'update', 'step': 6, 'writes': {'Node1': {'count': -3, 'lnode': 'node_1', 'scratch': 'hello'}}}, created_at='2025-01-06T19:16:08.664143+00:00', parent_config={'configurable': {'thread_id': '2', 'thread_ts': '1efcc617-1f54-6bf2-8005-15f572acfc33'}}) 

# StateSnapshot(values={'lnode': 'node_1', 'scratch': 'hello', 'count': 1}, next=('Node1',), config={'configurable': {'thread_id': '2', 'thread_ts': '1efcc617-1f54-6bf2-8005-15f572acfc33'}}, metadata={'source': 'update', 'step': 5, 'writes': {'Node2': {'count': -3, 'lnode': 'node_1', 'scratch': 'hello'}}}, created_at='2025-01-06T19:07:16.756980+00:00', parent_config={'configurable': {'thread_id': '2', 'thread_ts': '1efcc5ff-0eac-6c10-8004-8a83ce78de26'}}) 

# StateSnapshot(values={'lnode': 'node_2', 'scratch': 'hi', 'count': 4}, next=(), config={'configurable': {'thread_id': '2', 'thread_ts': '1efcc5ff-0eac-6c10-8004-8a83ce78de26'}}, metadata={'source': 'loop', 'step': 4, 'writes': {'Node2': {'count': 1, 'lnode': 'node_2'}}}, created_at='2025-01-06T18:56:30.765356+00:00', parent_config={'configurable': {'thread_id': '2', 'thread_ts': '1efcc5ff-0ea7-6f1e-8003-967364705237'}})

invoke will run from the current state if not given a particular thread_ts:
graph.invoke(None,thread2)
# node2, count:-2
# node1, count:-1
# node2, count:0
# node1, count:1
# node2, count:2

# {'lnode': 'node_2', 'scratch': 'hello', 'count': 3}
for state in graph.get_state_history(thread2):
    print(state,"\n")
# StateSnapshot(values={'lnode': 'node_2', 'scratch': 'hello', 'count': 3}, next=(), config={'configurable': {'thread_id': '2', 'thread_ts': '1efcc62e-bb0b-6e2e-800b-7f110e3dd0f9'}}, metadata={'source': 'loop', 'step': 11, 'writes': {'Node2': {'count': 1, 'lnode': 'node_2'}}}, created_at='2025-01-06T19:17:50.486471+00:00', parent_config={'configurable': {'thread_id': '2', 'thread_ts': '1efcc62e-bb08-63e6-800a-32cfe87ffcef'}}) 

# StateSnapshot(values={'lnode': 'node_1', 'scratch': 'hello', 'count': 2}, next=('Node2',), config={'configurable': {'thread_id': '2', 'thread_ts': '1efcc62e-bb08-63e6-800a-32cfe87ffcef'}}, metadata={'source': 'loop', 'step': 10, 'writes': {'Node1': {'count': 1, 'lnode': 'node_1'}}}, created_at='2025-01-06T19:17:50.484979+00:00', parent_config={'configurable': {'thread_id': '2', 'thread_ts': '1efcc62e-bb06-63af-8009-1e9607216f10'}}) 

# StateSnapshot(values={'lnode': 'node_2', 'scratch': 'hello', 'count': 1}, next=('Node1',), config={'configurable': {'thread_id': '2', 'thread_ts': '1efcc62e-bb06-63af-8009-1e9607216f10'}}, metadata={'source': 'loop', 'step': 9, 'writes': {'Node2': {'count': 1, 'lnode': 'node_2'}}}, created_at='2025-01-06T19:17:50.484151+00:00', parent_config={'configurable': {'thread_id': '2', 'thread_ts': '1efcc62e-bb00-6906-8008-01ce5f55f079'}}) 

# StateSnapshot(values={'lnode': 'node_1', 'scratch': 'hello', 'count': 0}, next=('Node2',), config={'configurable': {'thread_id': '2', 'thread_ts': '1efcc62e-bb00-6906-8008-01ce5f55f079'}}, metadata={'source': 'loop', 'step': 8, 'writes': {'Node1': {'count': 1, 'lnode': 'node_1'}}}, created_at='2025-01-06T19:17:50.481824+00:00', parent_config={'configurable': {'thread_id': '2', 'thread_ts': '1efcc62e-bafd-61fb-8007-a016c6ffff1e'}}) 

# StateSnapshot(values={'lnode': 'node_2', 'scratch': 'hello', 'count': -1}, next=('Node1',), config={'configurable': {'thread_id': '2', 'thread_ts': '1efcc62e-bafd-61fb-8007-a016c6ffff1e'}}, metadata={'source': 'loop', 'step': 7, 'writes': {'Node2': {'count': 1, 'lnode': 'node_2'}}}, created_at='2025-01-06T19:17:50.480416+00:00', parent_config={'configurable': {'thread_id': '2', 'thread_ts': '1efcc62a-effe-63cf-8006-ebe2d6f52a84'}}) 

# StateSnapshot(values={'lnode': 'node_1', 'scratch': 'hello', 'count': -2}, next=('Node2',), config={'configurable': {'thread_id': '2', 'thread_ts': '1efcc62a-effe-63cf-8006-ebe2d6f52a84'}}, metadata={'source': 'update', 'step': 6, 'writes': {'Node1': {'count': -3, 'lnode': 'node_1', 'scratch': 'hello'}}}, created_at='2025-01-06T19:16:08.664143+00:00', parent_config={'configurable': {'thread_id': '2', 'thread_ts': '1efcc617-1f54-6bf2-8005-15f572acfc33'}}) 

# StateSnapshot(values={'lnode': 'node_1', 'scratch': 'hello', 'count': 1}, next=('Node1',), config={'configurable': {'thread_id': '2', 'thread_ts': '1efcc617-1f54-6bf2-8005-15f572acfc33'}}, metadata={'source': 'update', 'step': 5, 'writes': {'Node2': {'count': -3, 'lnode': 'node_1', 'scratch': 'hello'}}}, created_at='2025-01-06T19:07:16.756980+00:00', parent_config={'configurable': {'thread_id': '2', 'thread_ts': '1efcc5ff-0eac-6c10-8004-8a83ce78de26'}}) 

# StateSnapshot(values={'lnode': 'node_2', 'scratch': 'hi', 'count': 4}, next=(), config={'configurable': {'thread_id': '2', 'thread_ts': '1efcc5ff-0eac-6c10-8004-8a83ce78de26'}}, metadata={'source': 'loop', 'step': 4, 'writes': {'Node2': {'count': 1, 'lnode': 'node_2'}}}, created_at='2025-01-06T18:56:30.765356+00:00', parent_config={'configurable': {'thread_id': '2', 'thread_ts': '1efcc5ff-0ea7-6f1e-8003-967364705237'}}) 

# StateSnapshot(values={'lnode': 'node_1', 'scratch': 'hi', 'count': 3}, next=('Node2',), config={'configurable': {'thread_id': '2', 'thread_ts': '1efcc5ff-0ea7-6f1e-8003-967364705237'}}, metadata={'source': 'loop', 'step': 3, 'writes': {'Node1': {'count': 1, 'lnode': 'node_1'}}}, created_at='2025-01-06T18:56:30.763384+00:00', parent_config={'configurable': {'thread_id': '2', 'thread_ts': '1efcc5ff-0ea4-6748-8002-3ee8b7f88f6f'}}) 

# StateSnapshot(values={'lnode': 'node_2', 'scratch': 'hi', 'count': 2}, next=('Node1',), config={'configurable': {'thread_id': '2', 'thread_ts': '1efcc5ff-0ea4-6748-8002-3ee8b7f88f6f'}}, metadata={'source': 'loop', 'step': 2, 'writes': {'Node2': {'count': 1, 'lnode': 'node_2'}}}, created_at='2025-01-06T18:56:30.761918+00:00', parent_config={'configurable': {'thread_id': '2', 'thread_ts': '1efcc5ff-0ea0-6e88-8001-ec0e11465a39'}}) 

# StateSnapshot(values={'lnode': 'node_1', 'scratch': 'hi', 'count': 1}, next=('Node2',), config={'configurable': {'thread_id': '2', 'thread_ts': '1efcc5ff-0ea0-6e88-8001-ec0e11465a39'}}, metadata={'source': 'loop', 'step': 1, 'writes': {'Node1': {'count': 1, 'lnode': 'node_1'}}}, created_at='2025-01-06T18:56:30.760499+00:00', parent_config={'configurable': {'thread_id': '2', 'thread_ts': '1efcc5ff-0e9c-6fa4-8000-749eeaf378c0'}}) 

# StateSnapshot(values={'scratch': 'hi', 'count': 0}, next=('Node1',), config={'configurable': {'thread_id': '2', 'thread_ts': '1efcc5ff-0e9c-6fa4-8000-749eeaf378c0'}}, metadata={'source': 'loop', 'step': 0, 'writes': None}, created_at='2025-01-06T18:56:30.758890+00:00', parent_config={'configurable': {'thread_id': '2', 'thread_ts': '1efcc5ff-0e99-6fce-bfff-54f64ddc5148'}}) 

# StateSnapshot(values={'count': 0}, next=('__start__',), config={'configurable': {'thread_id': '2', 'thread_ts': '1efcc5ff-0e99-6fce-bfff-54f64ddc5148'}}, metadata={'source': 'input', 'step': -1, 'writes': {'count': 0, 'scratch': 'hi'}}, created_at='2025-01-06T18:56:30.757665+00:00', parent_config=None) 
