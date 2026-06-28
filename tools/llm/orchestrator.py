import json
import os
import time
from typing import Dict, Any, Generator

from tools.context_management.context_handler import (
    read_context, write_context, append_to_chat_history, 
    update_context_from_llm
)
from tools.script_executor.sandbox import run_script_safely
from tools.llm.llm_client import generate_text, strip_json_fences

def process_user_message(message: str, paths: dict) -> dict:
    context = read_context(paths["context"])
    dataset_metadata = read_context(paths["dataset_metadata"])
    chat_history = read_context(paths["chat_history"])
    
    prompt = f"""You are a data analysis copilot orchestrator. Analyze the user's message and determine what information should be stored.

CURRENT CONTEXT:
{json.dumps(context, indent=2)}

DATASET METADATA:
{json.dumps(dataset_metadata, indent=2)}

RECENT CHAT HISTORY:
{json.dumps(chat_history.get('messages', [])[-5:], indent=2)}

USER MESSAGE:
{message}

TASK: Analyze this message and respond with a JSON object containing:
1. "response" - Your conversational response to the user
2. "context_update" - Any important information to add to context (company info, goals, constraints, etc.) or null if nothing to add
3. "needs_analysis" - Boolean indicating if this requires data analysis

Example output:
{{
  "response": "I understand you want to analyze sales trends. I'll help you with that.",
  "context_update": "User wants to analyze quarterly sales trends with focus on regional performance",
  "needs_analysis": true
}}

Respond with ONLY valid JSON, no additional text."""

    try:
        response_text = strip_json_fences(generate_text(prompt))
        result = json.loads(response_text)
        
        if result.get("context_update"):
            update_context_from_llm(
                paths["context"], 
                result["context_update"],
                source="user_chat" 
            )
        
        append_to_chat_history(paths["chat_history"], {
            "role": "user",
            "message": message,
            "timestamp": int(time.time())
        })
        
        append_to_chat_history(paths["chat_history"], {
            "role": "assistant",
            "message": result.get("response", ""),
            "context_update": result.get("context_update"),
            "timestamp": int(time.time())
        })
        
        return {
            "response": result.get("response", "I'm processing your message."),
            "needs_analysis": result.get("needs_analysis", False),
            "context_updated": result.get("context_update") is not None
        }
        
    except Exception as e:
        error_msg = f"Failed to process message: {str(e)}"
        print(f"ERROR in process_user_message: {error_msg}")
        append_to_chat_history(paths["chat_history"], {
            "role": "error",
            "message": error_msg,
            "timestamp": int(time.time())
        })
        return {
            "response": "I encountered an error processing your message. Please try again.",
            "error": error_msg,
            "needs_analysis": False
        }

def cotas_generate_insights(paths: dict, user_goal: str, max_loops: int = 15) -> Generator[str, None, None]:
    context = read_context(paths["context"])
    dataset_metadata = read_context(paths["dataset_metadata"])
    
    cotas_log = {"goal": user_goal, "steps": [], "start_time": int(time.time())}
    
    last_output = f"User Goal: {user_goal}"
    step = 0
    
    yield json.dumps({"type": "start", "message": "Starting CoTAS analysis...", "goal": user_goal})
    
    while step < max_loops:
        step += 1
        
        decision_prompt = f"""You are an autonomous data analysis agent using CoTAS methodology (Thought-Action-Search).s

ANALYSIS GOAL:
{user_goal}

CURRENT CONTEXT:
{json.dumps(context, indent=2)}

AVAILABLE DATASETS:
{json.dumps(dataset_metadata, indent=2)}

PREVIOUS OUTPUT:
{last_output}

COMPLETED STEPS: {step - 1}/{max_loops}

YOUR TASK:
Decide the next action to move towards the goal. Choose ONE:

1. THINK - Reasoning about what to do next, planning, or analyzing results
   Use when: You need to plan, reason about findings, or synthesize information

2. ACT - Execute Python code to analyze data, create visualizations, or compute results
   Use when: You need to process data, generate insights, or create outputs
   IMPORTANT: Code must be complete and runnable. Include all imports.
   Available data paths: storage/<session_id>/datasets/<filename>

3. DONE - Analysis complete, ready to provide final insights
   Use when: Goal is achieved and you have comprehensive findings

Respond with ONLY valid JSON in this EXACT format:

{{
  "action": "THINK",
  "content": "Your reasoning here",
  "context_update": "Important insight to remember" or null
}}

OR

{{
  "action": "ACT",
  "content": "import pandas as pd\\nimport matplotlib.pyplot as plt\\n# Complete Python code",
  "context_update": "What this analysis aims to discover" or null
}}

OR

{{
  "action": "DONE",
  "content": "Final comprehensive insights and findings",
  "context_update": null
}}

No additional text. Only valid JSON."""

        try:
            response_text = strip_json_fences(generate_text(decision_prompt))
            decision = json.loads(response_text)
            
            if "action" not in decision or "content" not in decision:
                raise ValueError("Missing required fields in decision")
                
        except Exception as e:
            error_entry = {
                "step": step,
                "action": "ERROR",
                "error": f"Failed to parse decision: {str(e)}",
                "timestamp": int(time.time())
            }
            cotas_log["steps"].append(error_entry)
            write_context(paths["cotas_log"], cotas_log)
            
            yield json.dumps({
                "type": "error",
                "step": step,
                "message": f"Decision parsing failed: {str(e)}"
            })
            break
        
        action = decision.get("action", "THINK").upper()
        content = decision.get("content", "")
        context_update = decision.get("context_update")
        
        if context_update:
            update_context_from_llm(paths["context"], context_update, f"CoTAS Step {step}")
            context = read_context(paths["context"])
        
        timestamp = int(time.time())
        
        if action == "THINK":
            entry = {
                "step": step,
                "action": "THINK",
                "content": content,
                "context_update": context_update,
                "timestamp": timestamp
            }
            cotas_log["steps"].append(entry)
            last_output = content
            
            yield json.dumps({
                "type": "think",
                "step": step,
                "content": content,
                "context_updated": context_update is not None
            })
        
        elif action == "ACT":
            yield json.dumps({
                "type": "act_start",
                "step": step,
                "message": "Executing code..."
            })
            
            script_filename = f"step_{step:03d}_{timestamp}.py"
            script_path = os.path.join(paths["scripts"], script_filename)
            
            try:
                with open(script_path, "w", encoding="utf-8") as f:
                    f.write(content)
            except Exception as e:
                yield json.dumps({
                    "type": "error",
                    "step": step,
                    "message": f"Failed to save script: {str(e)}"
                })
                continue
            
            try:
                session_id = context.get("session_id", "")
                modified_code = content.replace(
                    "storage/<session_id>/datasets/",
                    f"storage/{session_id}/datasets/"
                )
                
                exec_result = run_script_safely(modified_code, timeout=30)
                stdout = exec_result.get("stdout", "")
                stderr = exec_result.get("stderr", "")
                
            except Exception as e:
                stdout = ""
                stderr = f"Execution failed: {str(e)}"
            
            result_filename = f"step_{step:03d}_{timestamp}.txt"
            result_path = os.path.join(paths["results"], result_filename)
            
            with open(result_path, "w", encoding="utf-8") as f:
                f.write(f"=== STDOUT ===\n{stdout}\n\n=== STDERR ===\n{stderr}")
            
            entry = {
                "step": step,
                "action": "ACT",
                "script": script_filename,
                "result": result_filename,
                "stdout": stdout[:1000], 
                "stderr": stderr[:1000],
                "context_update": context_update,
                "timestamp": timestamp
            }
            cotas_log["steps"].append(entry)
            
            insight_prompt = f"""Analyze these code execution results:

CODE:
{content}

OUTPUT:
{stdout if stdout else "(no output)"}

ERRORS:
{stderr if stderr else "(no errors)"}

Provide a brief insight about:
1. What was accomplished
2. Key findings (if any)
3. What should happen next

Respond with only the insight text, no formatting."""

            try:
                insight = generate_text(insight_prompt, max_tokens=512).strip()
            except Exception:
                insight = stdout or stderr or "Execution completed"
            
            last_output = insight
            
            yield json.dumps({
                "type": "act_complete",
                "step": step,
                "stdout": stdout,
                "stderr": stderr,
                "insight": insight,
                "context_updated": context_update is not None
            })
        
        elif action == "DONE":
            entry = {
                "step": step,
                "action": "DONE",
                "final_insight": content,
                "timestamp": timestamp
            }
            cotas_log["steps"].append(entry)
            cotas_log["end_time"] = timestamp
            cotas_log["completed"] = True
            
            write_context(paths["cotas_log"], cotas_log)
            
            final_insight_path = os.path.join(paths["session"], "final_insight.txt")
            with open(final_insight_path, "w", encoding="utf-8") as f:
                f.write(content)
            
            yield json.dumps({
                "type": "done",
                "step": step,
                "final_insight": content
            })
            break
        
        else:
            yield json.dumps({
                "type": "error",
                "step": step,
                "message": f"Unknown action: {action}"
            })
            break
    
    if not cotas_log.get("completed"):
        cotas_log["end_time"] = int(time.time())
        cotas_log["completed"] = False
        cotas_log["reason"] = "Max loops reached"
    
    write_context(paths["cotas_log"], cotas_log)
    
    yield json.dumps({
        "type": "complete",
        "total_steps": step,
        "log_saved": True
    })
