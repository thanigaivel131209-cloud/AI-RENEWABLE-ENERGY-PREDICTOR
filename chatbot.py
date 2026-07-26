def get_bot_response(user_input):

    user_input = user_input.lower()

    if "solar" in user_input:
        return """☀️ Solar energy is electricity produced using sunlight.
It works best when the weather is sunny and cloud cover is low."""

    elif "wind" in user_input:
        return """🌬️ Wind energy is produced using wind turbines.
Higher wind speed generally increases power generation."""

    elif "renewable" in user_input:
        return """♻️ Renewable energy comes from natural sources such as
Solar, Wind, Hydro, Biomass and Geothermal."""

    elif "weather" in user_input:
        return """🌦️ This application uses live weather data
to predict renewable energy output."""

    elif "temperature" in user_input:
        return "🌡️ Temperature affects the efficiency of solar panels."

    elif "humidity" in user_input:
        return "💧 High humidity can slightly reduce solar efficiency."

    elif "cloud" in user_input:
        return "☁️ More cloud cover usually means lower solar energy generation."

    elif "hello" in user_input or "hi" in user_input:
        return "👋 Hello! I am your Renewable Energy AI Assistant."

    elif "thanks" in user_input:
        return "😊 You're welcome! Happy to help."

    else:
        return """🤖 I can answer questions about:
• Solar Energy
• Wind Energy
• Renewable Energy
• Weather
• Temperature
• Humidity
• Cloud Cover"""