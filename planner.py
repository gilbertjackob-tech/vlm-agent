import json
import os

class TaskPlanner:
    def __init__(self, kb_path="knowledge_base.json"):
        if os.path.exists(kb_path):
            with open(kb_path, 'r') as f:
                self.kb = json.load(f)
            self.entities = self.kb.get("entities", {})
            self.skills = self.kb.get("skills", {})
            print(f"Knowledge Base v{self.kb.get('version', 'Unknown')} Loaded.")
        else:
            print(f"Error: Could not find '{kb_path}'. Make sure the file exists.")
            self.entities = {}
            self.skills = {}

    def compile_plan(self, commands):
        execution_plan = []
        
        for cmd in commands:
            cmd_lower = cmd.lower().strip()
            
            # --- SKILL: OPEN APP ---
            if cmd_lower.startswith("open "):
                app_name = cmd_lower.replace("open ", "").strip()
                
                if app_name in self.entities:
                    app = self.entities[app_name]
                    print(f"Planner: Compiling routine to open '{app_name}'...")
                    
                    for step in self.skills["open_app"]["execution"]:
                        if step[0] == "click":
                            execution_plan.append(("click", app["visual_prompts"], step[2], app["expected_zones"][0]))
                        elif step[0] == "focus_window":
                            execution_plan.append(("focus_window", app["os_window_title"]))
                        else:
                            execution_plan.append(tuple(step))
                else:
                    print(f"Error: '{app_name}' is not in the Knowledge Base.")

            # --- SKILL: PARSE SCREEN (The new dynamic step!) ---
            elif cmd_lower in ["parse screen", "scan screen", "look around"]:
                print("Planner: Adding visual parsing step...")
                execution_plan.append(("parse_ui", ""))

            # --- SKILL: YOUTUBE SEARCH ---
            elif cmd_lower.startswith("search youtube for "):
                query = cmd[19:].strip() 
                print(f"Planner: Compiling routine to search YouTube for '{query}'...")
                
                for step in self.skills["youtube_search"]["execution"]:
                    if step[0] == "type" and step[1] == "{query}":
                        execution_plan.append(("type", query))
                    elif step[0] == "click_structured":
                        filters = step[1]
                        fallback = step[2] if len(step) > 2 else None
                        zone = step[3] if len(step) > 3 else None
                        execution_plan.append(("click_structured", filters, fallback, zone))
                    else:
                        execution_plan.append(tuple(step))
                        
            # --- SKILL: RELEASE VISION ---
            elif cmd_lower in ["release vision", "close app", "clear focus"]:
                execution_plan.append(("clear_window", ""))
                
            else:
                print(f"Planner Warning: I don't understand the command '{cmd}'")
                
        return execution_plan
