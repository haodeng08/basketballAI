import gradio as gr
import random
from typing import Dict, List
import datetime
import json
import os
from anthropic import Anthropic

# Install required packages (run this once)
try:
    import anthropic
except ImportError:
    print("Installing Anthropic package...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "anthropic"])
    import anthropic
    print("Anthropic package installed successfully")

# Google Drive Integration
try:
    from google.colab import drive
    import io
    COLAB_ENV = True
    print("Google Colab environment detected")
except ImportError:
    COLAB_ENV = False
    print("Running in local environment - Google Drive features limited")

# Database setup for Google Drive
GDRIVE_PATH = "/content/drive/My Drive/BasketballAI/" if COLAB_ENV else "./basketball_data/"
DB_FILE = "basketball_db.json"

# Integrate our model into our web app
class ClaudeIntegration:
    """Handles Claude API integration for basketball coaching"""

    def __init__(self, api_key: str):
        self.client = Anthropic(api_key=api_key)
        self.conversation_history = {}

        # Generel basketball context for model
        self.basketball_context = """
        You are an expert basketball coach AI assistant. Your role is to provide professional basketball training advice.

        Key responsibilities:
        - Provide specific drills, exercises, and training routines
        - Tailor advice to player position (Point Guard, Shooting Guard, Small Forward, Power Forward, Center)
        - Adjust recommendations based on skill level (beginner, intermediate, advanced)
        - Reference specific NBA players as examples when relevant
        - Focus on fundamentals, proper technique, and skill development
        - Be encouraging, motivational, and practical
        - Structure responses with clear actionable steps
        - Include duration estimates for drills when appropriate

        Basketball Knowledge Base:
        - Point Guards: Focus on ball handling, passing, court vision, leadership
        - Shooting Guards: Emphasize shooting form, footwork, cutting, perimeter defense
        - Small Forwards: Versatility, inside-outside game, rebounding, defensive switching
        - Power Forwards: Post moves, mid-range shooting, rebounding, interior defense
        - Centers: Post play, shot blocking, rebounding, interior presence

        Always provide accurate and professional advice that players can implement right away.
        """

    def get_claude_response(self, user_id: str, question: str, user_context: Dict = None) -> str:
        """Get response from Claude with basketball context and user info"""
        try:
            # Initialize conversation history for new users
            if user_id not in self.conversation_history:
                self.conversation_history[user_id] = []

            # Create enhanced prompt with user context
            enhanced_question = self._create_contextual_prompt(question, user_context)

            # Prepare messages for Claude
            messages = [{"role": "user", "content": self.basketball_context}]

            # Add conversation history (last 5 exchanges to keep context manageable)
            messages.extend(self.conversation_history[user_id][-10:])

            # Add current question
            messages.append({"role": "user", "content": enhanced_question})

            # Get Claude's response
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=1000,
                temperature=0.7,
                messages=messages
            )

            bot_reply = response.content[0].text

            # Update conversation history
            self.conversation_history[user_id].extend([
                {"role": "user", "content": enhanced_question},
                {"role": "assistant", "content": bot_reply}
            ])

            # Keep conversation history manageable (last 10 messages)
            if len(self.conversation_history[user_id]) > 10:
                self.conversation_history[user_id] = self.conversation_history[user_id][-10:]

            return bot_reply

        except Exception as e:
            print(f"Claude API Error: {e}")
            return f"I'm having trouble connecting to my AI coaching system right now. Error: {str(e)}"

    def _create_contextual_prompt(self, question: str, user_context: Dict = None) -> str:
        """Create an enhanced prompt with user context"""
        if not user_context:
            return question

        context_parts = []

        if user_context.get('position'):
            context_parts.append(f"Position: {user_context['position']}")

        if user_context.get('skill_level'):
            context_parts.append(f"Skill Level: {user_context['skill_level']}")

        if user_context.get('training_history'):
            recent_sessions = user_context['training_history'][-3:]  # Last 3 sessions
            if recent_sessions:
                context_parts.append(f"Recent Training: {', '.join(recent_sessions)}")

        if context_parts:
            context_string = " | ".join(context_parts)
            return f"[Player Context: {context_string}]\n\nQuestion: {question}"

        return question

# Data stored into my google drive as json file
GDRIVE_PATH = "/content/drive/My Drive/BasketballAI/"
DB_FILE = "basketball_db.json"
MODELS_DIR = "models/"

class DataManager:
    """Handles all data operations for the basketball AI"""

    def __init__(self):
        self.drive_mounted = False
        self.base_path = GDRIVE_PATH

        if COLAB_ENV:
            self.mount_drive()
        self.setup_directories()

    def mount_drive(self):
        """Mount Google Drive in Colab"""
        try:
            drive.mount('/content/drive')
            self.drive_mounted = True
            print("Google Drive mounted successfully")
        except Exception as e:
            print(f"Failed to mount Google Drive: {e}")
            self.drive_mounted = False

    def setup_directories(self):
        """Create necessary directories"""
        try:
            os.makedirs(self.base_path, exist_ok=True)
            print(f"Directory structure created at: {self.base_path}")
        except Exception as e:
            print(f"Failed to create directories: {e}")

    def save_json(self, data: dict, filename: str):
        """Save JSON data"""
        try:
            filepath = os.path.join(self.base_path, filename)
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Saved {filename}")
        except Exception as e:
            print(f"Failed to save {filename}: {e}")

    def load_json(self, filename: str) -> dict:
        """Load JSON data"""
        try:
            filepath = os.path.join(self.base_path, filename)
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    data = json.load(f)
                print(f"Loaded {filename}")
                return data
            else:
                print(f"{filename} not found, creating new file")
                return {}
        except Exception as e:
            print(f"Failed to load {filename}: {e}")
            return {}

class BasketballDatabase:
    """Database for user management and training history"""

    def __init__(self):
        self.data_manager = DataManager()
        self.db = self._initialize_db()

    # Initialize the database and its structure
    def _initialize_db(self) -> Dict:
        """Initialize or load the database"""
        existing_db = self.data_manager.load_json(DB_FILE)

        if existing_db:
            print("Loaded existing database")
            return existing_db

        print("Creating new database structure")
        return {
            "users": {},
            "session_stats": {
                "total_sessions": 0,
                "unique_users": 0,
                "last_updated": str(datetime.datetime.now())
            }
        }

    def save(self):
        """Save the database"""
        try:
            self.db["session_stats"]["last_updated"] = str(datetime.datetime.now())
            self.data_manager.save_json(self.db, DB_FILE)
        except Exception as e:
            print(f"Error saving database: {e}")

    def get_user(self, user_id: str) -> Dict:
        """Get or create a user profile"""
        if user_id not in self.db["users"]:
            self.db["users"][user_id] = {
                "position": "",
                "skill_level": "beginner",
                "created_at": str(datetime.datetime.now()),
                "last_session": "",
                "training_history": [],
                "total_sessions": 0,
                "preferences": {}
            }
            self.db["session_stats"]["unique_users"] = len(self.db["users"])
        return self.db["users"][user_id]

    def update_user(self, user_id: str, updates: Dict):
        """Update user information"""
        try:
            user = self.get_user(user_id)
            user.update(updates)
            user["last_session"] = str(datetime.datetime.now())
            self.save()
        except Exception as e:
            print(f"Error updating user: {e}")

    def add_training_session(self, user_id: str, session_summary: str):
        """Add a training session to user history"""
        user = self.get_user(user_id)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        user["training_history"].append(f"{timestamp}: {session_summary}")
        user["total_sessions"] += 1

        # Keep only last 10 sessions to manage storage
        if len(user["training_history"]) > 10:
            user["training_history"] = user["training_history"][-10:]

        self.db["session_stats"]["total_sessions"] += 1
        self.save()

class BasketballCoach:
    """Basketball coach that uses Claude AI"""

    def __init__(self, database: BasketballDatabase, claude_api_key: str = None):
        self.db = database
        self.claude = ClaudeIntegration(claude_api_key) if claude_api_key else None

    # Generative response based on the user input and profile
    def get_response(self, user_id: str, position: str, skill_level: str, question: str) -> str:
        """Generate response using Claude AI with user context"""
        try:
            user = self.db.get_user(user_id)

            # Update user profile if provided
            if position or skill_level:
                updates = {}
                if position:
                    updates["position"] = position
                if skill_level:
                    updates["skill_level"] = skill_level
                self.db.update_user(user_id, updates)
                user = self.db.get_user(user_id)  # Refresh user data

            # Check if the model is configuared
            if not self.claude:
                return "Claude AI is not configured. Please add your Anthropic API key to enable advanced coaching features."

            # Prepare user context for Claude
            user_context = {
                'position': user.get('position'),
                'skill_level': user.get('skill_level'),
                'training_history': user.get('training_history', []),
                'total_sessions': user.get('total_sessions', 0)
            }

            # Get Claude's response
            claude_response = self.claude.get_claude_response(user_id, question, user_context)

            # Add session to training history
            session_summary = f"Training session: {question[:50]}..." if len(question) > 50 else question
            self.db.add_training_session(user_id, session_summary)

            # Add metadata footer
            metadata = self._create_metadata(user, user_context)
            final_response = claude_response + metadata

            return final_response

        except Exception as e:
            print(f"Error generating response: {e}")
            return f"I encountered an error processing your request: {str(e)}"

    def _create_metadata(self, user: Dict, user_context: Dict) -> str:
        """Create metadata footer for responses"""
        storage_status = "Cloud Storage" if self.db.data_manager.drive_mounted else "Local Storage"
        total_sessions = user.get('total_sessions', 0)

        metadata = f"\n\n---\n"
        metadata += f"AI Coach model: Claude 3.5 Sonnet | "
        metadata += f"Sessions: {total_sessions} | "
        metadata += f"Storage: {storage_status}"

        if user_context.get('position'):
            metadata += f" | Position: {user_context['position']}"

        return metadata

# Initialize system and get API key from Claude
CLAUDE_API_KEY = "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"

print("Initializing Basketball AI with Claude...")
db = BasketballDatabase()
coach = BasketballCoach(db, CLAUDE_API_KEY)

# User interface initialization
def create_interface():
    with gr.Blocks(
        title="🏀 Basketball AI Coach",
        theme=gr.themes.Soft()
    ) as app:

        # Session state
        user_id = gr.State(lambda: str(random.randint(10000, 99999)))

        # Header
        gr.Markdown("""
        # 🏀 Basketball AI Coach
        Get personalized basketball training advice from an expert AI coach.
        """)



        # User profile section
        with gr.Accordion("👤 Player Profile", open=False):
            with gr.Row():
                position = gr.Dropdown(
                    choices=["", "Point Guard", "Shooting Guard", "Small Forward", "Power Forward", "Center"],
                    label="Primary Position",
                    value="",
                    interactive=True
                )
                skill_level = gr.Radio(
                    choices=["beginner", "intermediate", "advanced"],
                    label="Current Skill Level",
                    value="beginner"
                )

            gr.Markdown("*Profile information helps us provide more personalized coaching advice*")

        # Main chat interface
        chatbot = gr.Chatbot(
            label="🤖 Your AI Basketball Coach",
            bubble_full_width=False,
            height=500
        )

        with gr.Row():
            msg = gr.Textbox(
                placeholder="Ask me anything about basketball training, drills, techniques, or improvement tips...",
                label="Message",
                scale=4,
                container=False
            )
            send_btn = gr.Button("Send", scale=1, variant="primary")

        clear = gr.ClearButton([msg, chatbot], value="Clear Chat")

        # FAQ prompts
        gr.Examples(
            examples=[
                "Create a 30-minute shooting practice routine for me",
                "What are some defensive drills I can do alone?",
                "I'm struggling with my free throw consistency, help me improve",
                "Which NBA players should I study for my position?",
                "How do I work on my weak hand effectively?"
            ],
            inputs=msg,
            label="FAQ:"
        )

        # Training insight and features
        with gr.Accordion("Training Insights", open=False):
            training_display = gr.Textbox(
                value="""AI Basketball Coach Features:

• Personalized Coaching: Tailored advice based on your position and skill level
• Specific Drills: Get exact exercises with duration and technique tips
• NBA References: Learn from professional players as examples
• Progressive Training: Advice that grows with your skill development
• Conversation Memory: Claude remembers your previous questions for context

Your training history and preferences are automatically saved for personalized coaching!""",
                label="Features & Progress",
                lines=10,
                interactive=False
            )

        # Response function
        def respond(user_id, position, skill_level, message, chat_history):
            if not message.strip():
                return chat_history, ""

            try:
                # Get AI response
                bot_message = coach.get_response(user_id, position, skill_level, message)

                # Update chat history
                chat_history.append((message, bot_message))

                return chat_history, ""

            except Exception as e:
                print(f"Error in respond function: {e}")
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                chat_history.append((message, error_msg))
                return chat_history, ""

        # Event handlers
        msg.submit(
            respond,
            [user_id, position, skill_level, msg, chatbot],
            [chatbot, msg]
        )

        send_btn.click(
            respond,
            [user_id, position, skill_level, msg, chatbot],
            [chatbot, msg]
        )

        # Update training display when user interacts
        def update_training_info(user_id):
            try:
                user = db.get_user(user_id)
                total_sessions = user.get('total_sessions', 0)
                position = user.get('position', 'Not specified')
                skill_level = user.get('skill_level', 'beginner')

                recent_history = user.get('training_history', [])[-3:]  # Last 3 sessions
                history_text = "\n".join([f"• {session}" for session in recent_history]) if recent_history else "No sessions yet"

                return f""" Your Training Progress:

Total Sessions: {total_sessions}
Position: {position}
Skill Level: {skill_level.title()}
AI Coach: {'Claude 3.5 Sonnet Active' if coach.claude else 'Waiting for API key'}
Data Storage: {'Google Drive' if db.data_manager.drive_mounted else 'Local'}

Recent Training Sessions:
{history_text}

Keep practicing! Each session helps me understand your needs better."""
            except:
                return training_display.value

        # Update training info when chat changes
        chatbot.change(
            update_training_info,
            [user_id],
            [training_display]
        )

    return app

if __name__ == "__main__":
    try:
        print("Starting Basketball AI Coach...")
        print(f"Claude AI: {'Active' if coach.claude else 'API Key Required'}")
        print(f"Storage: {'Google Drive' if db.data_manager.drive_mounted else 'Local Mode'}")

        app = create_interface()
        app.launch(
            share=True,
            server_name="0.0.0.0",
            server_port=None
        )

    except Exception as e:
        print(f"Failed to launch app: {e}")
        print("\nMake sure you have the required packages:")
        print("pip install anthropic gradio")
