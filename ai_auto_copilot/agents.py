from llm import llm

def classify_intent(msg: str):
    return llm.invoke(
        f"Classify the student intent in one word (admission, fee, demo, schedule, other): {msg}"
    )

def generate_reply(msg: str):
    return llm.invoke(
        f"You are a polite admission counsellor. Reply to this student message and invite for demo class: {msg}"
    )

def followup():
    return llm.invoke(
        "Write a friendly follow-up message for a student who didn’t reply after 2 days."
    )
def send_user_response():
    return llm.invoke(
        "now you have got the respones from the llm now your responsibility is to make this message or convert this into a good human readable format so that can the user understands it better to what to do next "
    )
    print("Good now you can understand the things very well ans zoopark is now good so that you need not to worry about the things that you are trying to do now and one more thing that i wanted to tell you is that as long as you be here with tour ideas the ideas will keep flow like anything else that you never to in the future as well so now the thing is that whatever you have learnt untill now it's your own responsiblity to expose it to the real world and let the world know what you can do and what benifts that they can get from your experiance and your experties ")
    print("one more important thing that i wnated to tell is that in life whereever you go you have to be very very clear about your thoughts and your actions plans that what you are goona do in the new future so that you won't stuck at any point of time in the life and that is why i always wanted to tell you one thing as long as you are not good with the things you have to be vary curious in the life and don't leave at any point that without learning the things nothing would work in life and you know withour learning life no one will even consider you and all the fame recognization that you will be getting is not only through what you do but with the things that what you do")
    print("that is why i always keep telling that learning is the most valuable thing in the world that without it you cannot move forward atleast 1 step and you have to take a bow for that ")