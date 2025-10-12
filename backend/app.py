from flask import Flask, request, jsonify
from flask_cors import CORS
import uuid
import os
import json
import subprocess
import time
from datetime import datetime
import requests
import logging

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeminiAgentCreator:
    def __init__(self):
        self.gemini_api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

    def validate_api_key(self, api_key):
        """Validate Gemini API key by making a test request"""
        if not api_key or not api_key.startswith('AIza'):
            return False, "Invalid API key format"

        headers = {
            "x-goog-api-key": api_key,
            "Content-Type": "application/json"
        }

        test_payload = {
            "contents": [
                {
                    "parts": [{"text": "Hello"}]
                }
            ]
        }

        try:
            response = requests.post(
                self.gemini_api_url,
                headers=headers,
                json=test_payload,
                timeout=10
            )

            if response.status_code == 200:
                logger.info("API key validation successful")
                return True, "Valid API key"
            else:
                error_msg = response.json().get('error', {}).get('message', 'Unknown error')
                logger.warning(f"API key validation failed: {error_msg}")
                return False, f"API validation failed: {error_msg}"

        except requests.exceptions.RequestException as e:
            logger.error(f"API key validation error: {str(e)}")
            return False, f"Connection error: {str(e)}"

    def generate_functional_agent_code(self, prompt, api_key):
        """Generate a functional agent that handles user input and AI responses"""

        # For conversational/chat agents, generate a template that works
        if any(word in prompt.lower() for word in
               ['input', 'user', 'chat', 'talk', 'conversation', 'reply', 'respond']):
            return self.create_chat_agent_template(prompt), "success"

        # For other types, use Gemini to generate code
        return self.generate_agent_code_with_gemini(prompt, api_key)

    def create_chat_agent_template(self, prompt):
        """Create a working chat agent template"""

        code = f'''"""
AI-Enhanced Chat Agent
Original Prompt: {prompt}
Generated with Gemini AI on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
SECURITY: API keys are handled securely and not exposed in code
"""

import requests
import json
import logging
import os
import sys
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiChatAgent:
    """Interactive AI chat agent using Gemini API"""

    def __init__(self):
        self.api_key = self.get_api_key()
        self.api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
        self.conversation_history = []
        self.original_prompt = """{prompt}"""

        print("🤖 AI Chat Agent Initialized")
        print("💬 Ready to chat! Type 'exit' to quit.")

    def get_api_key(self):
        """Get API key from command line arguments or environment"""
        if len(sys.argv) > 1:
            return sys.argv[1]

        api_key = os.getenv('GEMINI_API_KEY')
        if api_key:
            return api_key

        return 'PLACEHOLDER_API_KEY'

    def make_gemini_request(self, user_message):
        """Send message to Gemini API and get response"""
        if self.api_key == 'PLACEHOLDER_API_KEY':
            return {{
                'error': True,
                'message': "❌ API key not configured. Please provide a valid Gemini API key."
            }}

        try:
            headers = {{
                "x-goog-api-key": self.api_key,
                "Content-Type": "application/json"
            }}

            # Build conversation context
            conversation_text = ""
            if self.conversation_history:
                conversation_text = "\\n\\nPrevious conversation:\\n"
                for entry in self.conversation_history[-5:]:  # Last 5 exchanges
                    conversation_text += f"User: {{entry['user']}}\\n"
                    conversation_text += f"Assistant: {{entry['assistant']}}\\n"

            full_prompt = f"You are a helpful AI assistant. Respond naturally and helpfully.{{conversation_text}}\\n\\nUser: {{user_message}}\\nAssistant:"

            payload = {{
                "contents": [
                    {{
                        "parts": [
                            {{
                                "text": full_prompt
                            }}
                        ]
                    }}
                ],
                "generationConfig": {{
                    "temperature": 0.7,
                    "maxOutputTokens": 1000,
                    "topP": 0.8,
                    "topK": 40
                }}
            }}

            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()

                if 'candidates' in result and result['candidates']:
                    ai_response = result['candidates'][0]['content']['parts'][0]['text']

                    # Add to conversation history
                    self.conversation_history.append({{
                        'user': user_message,
                        'assistant': ai_response,
                        'timestamp': datetime.now().isoformat()
                    }})

                    return {{
                        'error': False,
                        'message': ai_response,
                        'timestamp': datetime.now().isoformat()
                    }}
                else:
                    return {{
                        'error': True,
                        'message': "No response generated by Gemini"
                    }}
            else:
                error_data = response.json()
                error_msg = error_data.get('error', {{}}).get('message', f'HTTP {{response.status_code}}')
                return {{
                    'error': True,
                    'message': f"API Error: {{error_msg}}"
                }}

        except requests.exceptions.Timeout:
            return {{
                'error': True,
                'message': "Request timed out. Please try again."
            }}
        except requests.exceptions.RequestException as e:
            return {{
                'error': True,
                'message': f"Network error: {{str(e)}}"
            }}
        except Exception as e:
            logger.exception("Unexpected error in Gemini request")
            return {{
                'error': True,
                'message': f"Unexpected error: {{str(e)}}"
            }}

    def chat_loop(self):
        """Interactive chat loop"""
        print("\\n" + "="*50)
        print("🤖 AI CHAT AGENT")
        print("="*50)
        print("Start chatting! Type 'exit' to quit.")
        print()

        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print("\\n👋 Goodbye! Thanks for chatting!")
                    break

                # Show thinking indicator
                print("🤔 AI is thinking...")

                # Get AI response
                response = self.make_gemini_request(user_input)

                if response['error']:
                    print(f"\\n❌ Error: {{response['message']}}")
                else:
                    print(f"\\n🤖 AI: {{response['message']}}")

                print()  # Add spacing

            except KeyboardInterrupt:
                print("\\n\\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\\n❌ Unexpected error: {{e}}")
                break

    def execute_task(self):
        """Main execution method"""
        try:
            logger.info("Starting AI Chat Agent")

            # Check API key
            if self.api_key == 'PLACEHOLDER_API_KEY':
                print("\\n❌ Error: No Gemini API key provided!")
                print("Please run with: python agent.py YOUR_GEMINI_API_KEY")
                print("Or set GEMINI_API_KEY environment variable")
                return {{
                    'status': 'error',
                    'message': 'No API key provided'
                }}

            # Start chat
            self.chat_loop()

            # Return conversation summary
            return {{
                'status': 'completed',
                'conversations': len(self.conversation_history),
                'last_conversation': self.conversation_history[-1] if self.conversation_history else None,
                'total_time': datetime.now().isoformat()
            }}

        except Exception as e:
            logger.exception("Error in execute_task")
            return {{
                'status': 'error',
                'message': str(e)
            }}

if __name__ == "__main__":
    agent = GeminiChatAgent()
    result = agent.execute_task()
    print(f"\\n📊 Chat session result: {{result}}")
'''

        return code

    def generate_agent_code_with_gemini(self, prompt, api_key):
        """Generate agent code using Gemini API for non-chat tasks"""
        headers = {
            "x-goog-api-key": api_key,
            "Content-Type": "application/json"
        }

        system_prompt = f"""You are an expert Python developer. Create a complete, functional Python class based on this request:

"{prompt}"

Requirements:
1. Create a complete Python class with proper imports
2. Include comprehensive error handling and logging
3. Make it actually functional with real implementations - no placeholder code
4. Include a main execution method called execute_task()
5. Use appropriate libraries (requests, json, datetime, etc.)
6. Include detailed docstrings and comments
7. Make the code production-ready and robust
8. NEVER include actual API keys in the generated code - use placeholder text instead
9. If the task needs AI capabilities, include placeholder comments for API integration

Important: Return ONLY the Python code, no markdown formatting or explanations."""

        payload = {
            "contents": [
                {
                    "parts": [{"text": system_prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 3000,
                "topP": 0.8,
                "topK": 40
            }
        }

        try:
            response = requests.post(
                self.gemini_api_url,
                headers=headers,
                json=payload,
                timeout=45
            )

            if response.status_code == 200:
                result = response.json()

                if 'candidates' in result and result['candidates']:
                    generated_code = result['candidates'][0]['content']['parts'][0]['text']

                    # Clean up the code
                    if "```python" in generated_code:
                        parts = generated_code.split("```python")
                        if len(parts) > 1:
                            generated_code = parts[1].split("```")[0].strip()
                    elif "```" in generated_code:
                        parts = generated_code.split("```")
                        if len(parts) > 1:
                            generated_code = parts[1].strip()

                    # Sanitize code
                    generated_code = self.sanitize_generated_code(generated_code)

                    logger.info("Agent code generated successfully")
                    return generated_code, "success"

            else:
                error_data = response.json()
                error_msg = error_data.get('error', {}).get('message', 'Unknown error')
                logger.error(f"Gemini API error: {error_msg}")
                return None, f"Gemini API error: {error_msg}"

        except requests.exceptions.Timeout:
            return None, "Request timeout - please try again"
        except Exception as e:
            logger.error(f"Code generation error: {str(e)}")
            return None, f"Code generation failed: {str(e)}"

    def sanitize_generated_code(self, code):
        """Remove any actual API keys from generated code"""
        import re

        # Remove any actual API keys that start with common patterns
        api_key_patterns = [
            r'AIza[a-zA-Z0-9_-]+',  # Google API keys
            r'sk-[a-zA-Z0-9]+',  # OpenAI keys
            r'xoxb-[a-zA-Z0-9-]+',  # Slack tokens
        ]

        for pattern in api_key_patterns:
            code = re.sub(pattern, 'YOUR_API_KEY_HERE', code)

        return code

    def create_agent(self, prompt, api_key):
        """Create a new AI agent based on the given prompt"""
        try:
            # Check if this is a sentiment analysis request
            if any(term in prompt.lower() for term in ['sentiment', 'emotion', 'feelings', 'mood']):
                return self.create_sentiment_agent(prompt, api_key)
                
            # Generate agent code using Gemini
            agent_code, status = self.generate_agent_code_with_gemini(prompt, api_key)
            if agent_code is None:
                return None, status
                
            return {
                'code': agent_code,
                'model': 'gemini-2.0-flash',
                'capabilities': ['text_processing'],
                'description': f'AI agent for: {prompt[:100]}{"..." if len(prompt) > 100 else ""}'
            }, "success"
            
        except Exception as e:
            logger.error(f"Error creating agent: {str(e)}", exc_info=True)
            return None, f"Failed to create agent: {str(e)}"
            
    def create_sentiment_agent(self, prompt, api_key):
        """Create a sentiment analysis agent with proper error handling"""
        try:
            code = """import logging
import json
from datetime import datetime
import requests
from typing import Dict, Optional

class SentimentAnalyzer:
    \"\"\"A class for analyzing text sentiment using the Gemini API\"\"\"

    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url
        self.api_key = api_key
        self.logger = logging.getLogger(__name__)
        self.setup_logging()

    def setup_logging(self):
        \"\"\"Configure logging settings\"\"\"
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def analyze_sentiment(self, text: str) -> Optional[Dict]:
        \"\"\"Analyze text sentiment using Gemini API\"\"\"
        if not self.api_url or not self.api_key:
            self.logger.error(\"API configuration missing\")
            return None

        prompt = \"\"\"Analyze the sentiment of the following text and return a JSON response with:
        - sentiment (positive/neutral/negative)
        - confidence (0.0-1.0)
        - emotions (dictionary of emotion scores 0.0-1.0)
        \n        Text: \"\"\" + text

        try:
            response = requests.post(
                self.api_url,
                params={'key': self.api_key},
                json={
                    \"contents\": [{
                        \"parts\": [{\"text\": prompt}]
                    }]
                },
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            if 'candidates' in result and result['candidates']:
                response_text = result['candidates'][0]['content']['parts'][0].get('text', '')
                return self.parse_response(response_text)
                
        except Exception as e:
            self.logger.error(f\"API request failed: {str(e)}\")
            return None

    def parse_response(self, response_text: str) -> Dict:
        \"\"\"Parse the API response text into a structured format\"\"\"
        try:
            # Clean the response and extract JSON
            import re
            json_str = re.search(r'\\{.*\\}', response_text, re.DOTALL)
            if json_str:
                return json.loads(json_str.group(0))
            return {\"error\": \"Invalid response format\"}
        except Exception as e:
            self.logger.error(f\"Failed to parse response: {str(e)}\")
            return {\"error\": str(e)}

    def generate_report(self, sentiment_data: Dict) -> str:
        \"\"\"Generate a human-readable report from sentiment data\"\"\"
        if not sentiment_data or 'error' in sentiment_data:
            return \"Unable to analyze sentiment. Please try again later.\"
            
        report = [
            f\"Sentiment Analysis Report\",
            f\"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\",
            f\"Overall Sentiment: {sentiment_data.get('sentiment', 'unknown')}\",
            f\"Confidence: {sentiment_data.get('confidence', 0) * 100:.1f}%\"
        ]
        
        if 'emotions' in sentiment_data:
            report.append(\"\\nEmotion Scores:\")
            for emotion, score in sentiment_data['emotions'].items():
                report.append(f\"- {emotion.capitalize()}: {float(score) * 100:.1f}%\")
                
        return '\\n'.join(report)

# Example usage
if __name__ == \"__main__\":
    # Initialize with your API details
    analyzer = SentimentAnalyzer(
        api_url=\"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent\",
        api_key=\"YOUR_API_KEY\"  # Replace with actual API key
    )
    
    # Example analysis
    sample_text = \"I'm feeling really happy and excited about this new project!\"
    result = analyzer.analyze_sentiment(sample_text)
    print(analyzer.generate_report(result))
"""
            
            return {
                'code': code,
                'model': 'gemini-1.5-flash',
                'capabilities': ['sentiment_analysis', 'emotion_detection', 'text_processing'],
                'description': 'Advanced sentiment analysis agent using Gemini AI'
            }, "success"
            
        except Exception as e:
            logger.error(f"Error creating sentiment agent: {str(e)}", exc_info=True)
            return None, f"Failed to create sentiment analysis agent: {str(e)}"
agent_creator = GeminiAgentCreator()


@app.route('/', methods=['GET'])
def home():
    """API health check"""
    return jsonify({
        'message': 'Gemini AI Agent Creator is running!',
        'version': '2.1',
        'ai_model': 'Google Gemini 1.5 Flash',
        'features': 'Interactive chat agents with conversation memory',
        'capabilities': [
            'Interactive Chat Agents',
            'AI-Powered Conversations',
            'Real-time Responses',
            'Secure API Key Handling'
        ]
    })


@app.route('/create-agent', methods=['POST'])
def create_agent():
    """Create a new AI agent from user prompt"""
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        prompt = data.get('prompt', '').strip()
        gemini_api_key = data.get('gemini_api_key', '').strip()

        if not prompt:
            return jsonify({'error': 'Prompt is required'}), 400

        if not gemini_api_key:
            return jsonify({'error': 'Gemini API key is required'}), 400

        if not gemini_api_key.startswith('AIza'):
            return jsonify({'error': 'Invalid Gemini API key format. Keys should start with "AIza"'}), 400

        logger.info(f"Creating agent for prompt: {prompt[:50]}...")

        # Create agent using Gemini API
        agent_data, status = agent_creator.create_agent(prompt, gemini_api_key)

        if agent_data is None:
            return jsonify({'error': status}), 400

        # Generate unique agent ID
        agent_id = str(uuid.uuid4())

        # Create agent directory
        agent_dir = f"generated_agents/{agent_id}"
        os.makedirs(agent_dir, exist_ok=True)

        # Save agent code
        agent_file_path = f"{agent_dir}/agent.py"
        with open(agent_file_path, 'w', encoding='utf-8') as f:
            f.write(agent_data['code'])

        # Create metadata
        metadata = {
            'id': agent_id,
            'prompt': prompt,
            'model': agent_data['model'],
            'capabilities': agent_data['capabilities'],
            'description': agent_data['description'],
            'created_at': datetime.now().isoformat(),
            'status': 'created',
            'ai_powered': True,
            'interactive': True
        }

        # Save metadata
        with open(f"{agent_dir}/metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Agent {agent_id} created successfully")

        return jsonify({
            'agent_id': agent_id,
            'status': 'created',
            'prompt': prompt,
            'model': agent_data['model'],
            'capabilities': agent_data['capabilities'],
            'description': agent_data['description'],
            'created_at': metadata['created_at'],
            'code': agent_data['code'][:500] + '...' if len(agent_data['code']) > 500 else agent_data['code']
        })

    except Exception as e:
        logger.error(f"Agent creation error: {str(e)}")
        return jsonify({'error': f'Failed to create agent: {str(e)}'}), 500


@app.route('/execute-agent/<agent_id>', methods=['POST'])
def execute_agent(agent_id):
    """Execute an AI agent securely"""
    try:
        agent_dir = f"generated_agents/{agent_id}"
        agent_script = f"{agent_dir}/agent.py"

        if not os.path.exists(agent_script):
            return jsonify({'error': 'Agent not found'}), 404

        # Get API key from request for execution
        data = request.get_json()
        gemini_api_key = data.get('gemini_api_key', '') if data else ''

        if not gemini_api_key:
            return jsonify({'error': 'Gemini API key required for agent execution'}), 400

        logger.info(f"Executing agent {agent_id}")

        # Execute the agent with API key passed as argument
        result = subprocess.run(
            ['python', agent_script, gemini_api_key],
            cwd=agent_dir,
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes for interactive agents
        )

        execution_result = {
            'status': 'completed',
            'agent_id': agent_id,
            'return_code': result.returncode,
            'output': result.stdout,
            'error': result.stderr if result.stderr else None,
            'executed_at': datetime.now().isoformat(),
            'ai_powered': True,
            'interactive': True
        }

        if result.returncode == 0:
            logger.info(f"Agent {agent_id} executed successfully")
        else:
            logger.warning(f"Agent {agent_id} execution failed with return code {result.returncode}")

        return jsonify(execution_result)

    except subprocess.TimeoutExpired:
        logger.error(f"Agent {agent_id} execution timed out")
        return jsonify({
            'error': 'Agent execution timed out (5 minutes limit)',
            'agent_id': agent_id
        }), 408
    except Exception as e:
        logger.error(f"Agent execution error: {str(e)}")
        return jsonify({'error': f'Execution failed: {str(e)}'}), 500


@app.route('/agents', methods=['GET'])
def list_agents():
    """List all created agents"""
    try:
        agents_dir = "generated_agents"

        if not os.path.exists(agents_dir):
            return jsonify([])

        agents = []

        for agent_id in os.listdir(agents_dir):
            metadata_path = f"{agents_dir}/{agent_id}/metadata.json"

            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                        agents.append(metadata)
                except Exception as e:
                    logger.warning(f"Failed to load metadata for agent {agent_id}: {str(e)}")
                    continue

        # Sort by creation date (newest first)
        agents.sort(key=lambda x: x.get('created_at', ''), reverse=True)

        return jsonify(agents)

    except Exception as e:
        logger.error(f"Failed to list agents: {str(e)}")
        return jsonify({'error': f'Failed to list agents: {str(e)}'}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print("🚀 Starting Interactive AI Agent Creator...")
    print("🧠 Powered by Google Gemini 1.5 Flash")
    print("💬 Creating interactive chat agents with conversation memory")
    print("🔐 API keys are handled securely and never exposed")
    print("=" * 80)

    # Create agents directory if it doesn't exist
    os.makedirs('generated_agents', exist_ok=True)

    app.run(host='0.0.0.0', port=8001, debug=True)
