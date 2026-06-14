def get_bot_response(user_input):
    user_input = user_input.lower()

    if "solar" in user_input:
        return "Solar energy comes from sunlight converted into electricity using solar panels."
    
    elif "wind" in user_input:
        return "Wind energy is generated using wind turbines that convert wind into electrical power."
    
    elif "renewable" in user_input:
        return "Renewable energy comes from natural sources like sun, wind, and water."

    elif "hello" in user_input or "hi" in user_input:
        return "Hello! I am your AI Energy Assistant ⚡ Ask me anything about renewable energy."

    else:
        return "I can help with solar, wind, and renewable energy questions."