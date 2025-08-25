# # agents/llm_interface.py
# import ollama

# def local_llm(prompt: str) -> str:
#     """
#     Sends the prompt to a local LLM via Ollama.
#     Returns the model's response as text.
#     """
#     try:
#         response = ollama.chat(
#             model="deepseek-r1:1.5b",  # replace with your local model
#             messages=[{
#                 "role": "user", "content": prompt}]
#         )
#         # Extract content
#         return response['message']['content']

#     except Exception as e:
#         return f"[LLM Error] {str(e)}"


# def classify_message_small(prompt: str) -> str:
#     """
#     Lightweight classification for PUBLIC / PRIVATE / MENTAL_HEALTH.
#     Can be used in OrchestrationAgent to avoid calling big LLM every time.
#     """
#     try:
#         response = ollama.chat(
#             model="deepseek-r1:1.5b",  # smaller model for classification
#             messages=[{"role": "user", "content": prompt}]
#         )
#         return response['message']['content'].strip().upper()
#     except Exception as e:
#         return "PUBLIC"  # default safe category

# agents/llm_interface.py
import ollama

def local_llm(prompt: str) -> str:
    """
    Sends the prompt to a local LLM via Ollama.
    Returns the model's response as text without reasoning blocks.
    """
    try:
        # Wrap user prompt to instruct model to return only the final answer
        wrapped_prompt = f"""
        You are like a friend, assistant and more. you can conversation as a human as well as answering questions.
        

        Question: {prompt}
        """

        response = ollama.chat(
            model="deepseek-r1:1.5b",  
            messages=[{"role": "user", "content": wrapped_prompt}]
        )

        # Extract content and remove any <think> blocks
        result = response['message']['content']
        if "<think>" in result:
            result = result.split("</think>")[-1].strip()
        return result

    except Exception as e:
        return f"[LLM Error] {str(e)}"


def classify_message_small(prompt: str) -> str:
    """
    Lightweight classification for PUBLIC / PRIVATE / MENTAL_HEALTH.
    Returns only the category in uppercase.
    """
    try:
        wrapped_prompt = f"""
        Classify the following message into PUBLIC, PRIVATE, or MENTAL_HEALTH.
        Only respond with the label in uppercase. Do NOT provide explanations.

        Message: {prompt}
        """

        response = ollama.chat(
            model="deepseek-r1:1.5b",  # smaller model for classification
            messages=[{"role": "user", "content": wrapped_prompt}]
        )

        # Extract content and clean
        result = response['message']['content'].strip().upper()
        # In case the model returns extra text accidentally
        result = result.split()[0] if result else "PUBLIC"
        return result

    except Exception as e:
        return "PUBLIC"  # default safe category
