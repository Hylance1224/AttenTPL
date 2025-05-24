import openai
import os

os.environ["https_proxy"]="127.0.0.1:4780"


openai.api_key = "XXX"


def get_functions(description):
    gpt_response = openai.ChatCompletion.create(
        model="gpt-4.0-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content":
                "An app function refers to an essential function or a service that the app can provide for users."
                "There are some function examples: \"navigate to the destination\", \"photo editing with filters\"."
                "List functions from the following app "
                "(ensuring each function description does not exceed eight words)."
                "Do not extract statements related to updates or initiatives, such as \"Start using the app\", \"enjoy your app\""
                + description},
        ],
        temperature=0.2,
        max_tokens=500
    )
    functions = gpt_response['choices'][0]['message']['content']
    return functions


if __name__ == '__main__':
    description = ""
    functions = get_functions(description)
    print(functions)



