from langgraph.graph import StateGraph
from agents import classify_intent, generate_reply

class State(dict): pass

def build_graph():
    g = StateGraph(State)
    g.add_node("intent", lambda s: {"intent": classify_intent(s["msg"])})
    g.add_node("reply", lambda s: {"reply": generate_reply(s["msg"])})
    g.set_entry_point("intent")
    g.add_edge("intent", "reply")
    return g.compile()
