# comp9900_x-o-bot_preproData
how to pre-process the json file data

simply download my code, and store the file like below:
9900--webcms3json--messages.json
      |
      |
      raw2dict.py
      |
      |
      example.jpg

before you start it, you need to firstly pip install the "json" package

"cd" to the "9900" directory and then run the command "python3 raw2dict.py"

after it done, you can see a new json file "id2content.json" occur at your current directory

you can use it like the commands showed in example.jpg

"from raw2dict import load_dict"
"myDict = load_dict('id2content.json')"
"myDict['2691543']['message_body']"

"myDict" is a dictionary, you can find the message by their "mesg_id", in STRING format! Besides, in the message object, "mesg_id", "parent_id", "posted_by" and "resource_id" are in INTEGER format! And every first comment will not contain "parent_id"!

Feel free to use it!
