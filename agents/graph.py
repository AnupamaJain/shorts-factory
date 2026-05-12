import os
from dotenv import load_dotenv
load_dotenv()
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
import json
import yaml
import os

# Load Configuration
config_path = os.path.join(os.path.dirname(__file__), "..", "shorts_config.yaml")
with open(config_path, 'r') as f:
    SHORTS_CONFIG = yaml.safe_load(f)

# Local modules
from agents.rag import ShortsRAG
from agents.tools import render_video
from agents.social_media_agent import generate_instagram_metadata
from agents.publisher_agent import post_video_to_instagram

# DeepEval Integration
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models import DeepEvalBaseLLM

class GroqDeepEval(DeepEvalBaseLLM):
    def __init__(self, model):
        self.model = model
    def load_model(self): return self.model
    def generate(self, prompt: str) -> str:
        return self.model.invoke(prompt).content
    async def a_generate(self, prompt: str) -> str:
        return self.model.invoke(prompt).content
    def get_model_name(self): return "groq-llama3"

# 1. Define the State
class AgentState(TypedDict):
    topic: str
    research_data: str
    draft_script: str
    feedback: str  # Guardrail feedback
    evaluator_feedback: str # Quality feedback
    quality_score: int
    revision_count: int
    final_output: str
    compliance_passed: bool
    video_path: str
    social_media_caption: str
    thumbnail_path: str

# 2. Initialize Models
# Ensure GROQ_API_KEY is in your environment variables
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)
reviewer_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2) # lower temp for strict review

# Initialize RAG
rag = ShortsRAG()

# 3. Define Nodes
def researcher_node(state: AgentState):
    print("--- RESEARCHER ---")
    topic = state["topic"]
    
    try:
        retriever = rag.get_retriever(k=3)
        docs = retriever.invoke(topic)
        research_data = "\n\n".join([d.page_content for d in docs])
    except Exception as e:
        print(f"RAG not initialized or error: {e}. Proceeding without RAG.")
        research_data = "No specific historical context found."

    # Ask LLM to summarize research
    prompt = f"Topic: {topic}\nContext:\n{research_data}\n\nSummarize the key points to include in a short viral finance video."
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {"research_data": response.content}

def scriptwriter_node(state: AgentState):
    print("--- SCRIPTWRITER ---")
    topic = state["topic"]
    research = state.get("research_data", "")
    feedback = state.get("feedback", "")
    eval_feedback = state.get("evaluator_feedback", "")
    
    prompt = f"""
    Write a 60-second YouTube Short script about: {topic}.
    Use this research: {research}
    
    Ensure it has a strong hook (fear or greed). Keep sentences short and punchy.
    CRITICAL RULES FOR SCRIPT:
    1. Output ONLY the exact words that will be spoken by the voiceover.
    2. DO NOT include any timestamps (e.g., 0:00-0:05).
    3. DO NOT include any stage directions or audio cues (e.g., [Intro], (Music plays), Narrator:).
    4. DO NOT use markdown formatting like **bold** or *italics*.
    5. DO NOT include conversational filler like "Here is the script" or "Sure!". Just the script itself.
    """
    if feedback:
        prompt += f"\n\nCRITICAL FEEDBACK FROM COMPLIANCE REVIEW: {feedback}\nPlease fix these issues."
    if eval_feedback:
        prompt += f"\n\nCRITICAL FEEDBACK FROM VIRAL EVALUATOR: {eval_feedback}\nPlease improve the script's engagement based on this feedback."
        
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "draft_script": response.content, 
        "revision_count": state.get("revision_count", 0) + 1
    }

def evaluator_node(state: AgentState):
    print("--- QUALITY EVALUATOR (DEEPEVAL) ---")
    script = state["draft_script"]
    topic = state["topic"]
    
    try:
        # Define the custom G-Eval metric
        viral_metric = GEval(
            name="Viral Potential and Authority",
            criteria=SHORTS_CONFIG.get("evaluator", {}).get("criteria", "Determine if the script has a strong hook."),
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
            model=GroqDeepEval(reviewer_llm)
        )
        
        # Create the test case
        test_case = LLMTestCase(
            input=topic,
            actual_output=script
        )
        
        # Measure
        viral_metric.measure(test_case)
        
        # DeepEval returns score 0.0 to 1.0. Convert to 1-10.
        score = int(viral_metric.score * 10)
        feedback = viral_metric.reason
        
    except Exception as e:
        print(f"DeepEval failed: {e}")
        score = 5
        feedback = "Failed to run DeepEval programmatic tests."
        
    print(f"DeepEval GEval Score: {score}/10")
    print(f"DeepEval Reason: {feedback}")
    
    return {"quality_score": score, "evaluator_feedback": feedback}

def guardrail_node(state: AgentState):
    print("--- GUARDRAIL REVIEWER ---")
    script = state["draft_script"]
    
    prompt = f"""
    You are a strict financial compliance officer. Review this YouTube Short script:
    
    SCRIPT:
    {script}
    
    RULES:
    1. It MUST NOT guarantee any profits or returns.
    2. It MUST NOT give explicit financial advice (e.g., "Buy XYZ now").
    3. It MUST NOT sound like a scam.
    
    If it passes all rules, respond exactly with "PASS". 
    If it violates any rules, respond with "FAIL:" followed by the specific feedback on what to change.
    """
    
    response = reviewer_llm.invoke([HumanMessage(content=prompt)])
    review = response.content.strip()
    
    if review.startswith("FAIL"):
        return {"feedback": review, "compliance_passed": False}
    else:
        return {"feedback": "PASS", "compliance_passed": True}

def director_node(state: AgentState):
    print("--- DIRECTOR ---")
    script = state["draft_script"]
    topic = state["topic"]
    
    # 1. Ask LLM to pick an emotion and filename
    prompt = f"Given this script: '{script[:200]}...', pick ONE emotion_theme (neutral, aggressive, poetic, fear) and a filename (e.g. {topic.replace(' ', '_')}.mp4). Return JSON: {{\"emotion\": \"...\", \"filename\": \"...\"}}"
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        config = json.loads(response.content)
        emotion = config.get("emotion", "neutral")
        filename = config.get("filename", "agent_output.mp4")
    except:
        emotion = "neutral"
        filename = f"{topic.replace(' ', '_')}.mp4"

    # 2. Call the MCP Tool
    print(f"Director is calling render_video tool for {filename}...")
    render_video.invoke({"script": script, "emotion_theme": emotion, "output_filename": filename})
    
    state["video_path"] = f"outputs/{filename}"
    return state

def social_media_node(state: AgentState):
    print("--- SOCIAL MEDIA MANAGER ---")
    script = state["draft_script"]
    # Generate the Instagram post
    post_data = generate_instagram_metadata(script)
    return {
        "social_media_caption": post_data.get("full_post", ""),
        "thumbnail_path": post_data.get("thumbnail_path", "")
    }

def publisher_node(state: AgentState):
    print("--- PUBLISHER AGENT ---")
    auto_post = SHORTS_CONFIG.get("publisher", {}).get("auto_post_instagram", False)
    
    if auto_post:
        video_path = state.get("video_path")
        caption = state.get("social_media_caption", "")
        thumbnail_path = state.get("thumbnail_path", "")
        
        if video_path and caption:
            post_video_to_instagram(video_path, caption, thumbnail_path)
        else:
            print("⚠️ Missing video_path or caption. Skipping upload.")
    else:
        print("⏸️ Auto-posting is disabled in shorts_config.yaml. Skipping Instagram upload.")
        
    return state

# 5. Build the Graph
workflow = StateGraph(AgentState)

workflow.add_node("researcher", researcher_node)
workflow.add_node("scriptwriter", scriptwriter_node)
workflow.add_node("evaluator", evaluator_node)
workflow.add_node("guardrail", guardrail_node)
workflow.add_node("director", director_node)
workflow.add_node("social_media", social_media_node)
workflow.add_node("publisher", publisher_node)

workflow.set_entry_point("researcher")

# Edges
workflow.add_edge("researcher", "scriptwriter")
workflow.add_edge("scriptwriter", "evaluator")

# Conditional Routing from Evaluator
def check_quality(state: AgentState):
    pass_threshold = SHORTS_CONFIG.get("evaluator", {}).get("pass_threshold", 8)
    if state.get("quality_score", 0) >= pass_threshold or state.get("revision_count", 0) >= 3:
        return "guardrail"
    else:
        return "scriptwriter"

workflow.add_conditional_edges(
    "evaluator",
    check_quality,
    {
        "guardrail": "guardrail",
        "scriptwriter": "scriptwriter"
    }
)

# Conditional Routing from Guardrail
def check_compliance(state: AgentState):
    if state["compliance_passed"] or state.get("revision_count", 0) >= 4:
        return "director"
    else:
        return "scriptwriter"

workflow.add_conditional_edges(
    "guardrail",
    check_compliance,
    {
        "director": "director",
        "scriptwriter": "scriptwriter"
    }
)

workflow.add_edge("director", "social_media")
workflow.add_edge("social_media", "publisher")
workflow.add_edge("publisher", END)

# Compile
app = workflow.compile()

if __name__ == "__main__":
    import sys
    topic = "The dangers of zero day options"
    if len(sys.argv) > 1:
        topic = sys.argv[1]
        
    print(f"Starting Multi-Agent Pipeline for Topic: {topic}")
    initial_state = {"topic": topic, "revision_count": 0}
    
    # Run the graph
    for output in app.stream(initial_state):
        # stream yields a dict with the node name as key
        for key, value in output.items():
            print(f"Finished node: {key}")
            
    print("PIPELINE COMPLETE.")
