import preprocess as pre
import eval 


if __name__ == "__main__":
    text = {
        "text1" : "I enjoy listening to music in my free time.",
        "text2" : "Listening to music is something I enjoy doing in my spare time.",
    }
    nn = eval.simEngCheck()
    result = nn.forward(text["text1"],text["text2"])
    print(result)

"""
{
    "text1": "I enjoy listening to music in my free time.", 
    "text2": "Listening to music is something I enjoy doing in my spare time.",
    "similarity": 0.6645563244819641
}
"""
