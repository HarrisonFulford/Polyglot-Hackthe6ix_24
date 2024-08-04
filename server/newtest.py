import asyncio
import socket
import json
from translator import translate_text
import mongo
import re

compiled_t = ""

async def handle_client(reader, writer):
    addr = writer.get_extra_info('peername')
    print(f"Connected by {addr}")

    async def translate_title(data):
        print(data)
        msg = translate_text(data['msg'], target_language=data['targetLocale'])
        print(msg)

        asyncio.create_task(
            send_message(
                json.dumps({
                    'msg': msg,
                    'id': 'translate_title',
                    'targetLocale': data['targetLocale']
                })
            )
        )

    def split_list(lst, n):
        return [lst[i:i + n] for i in range(0, len(lst), n)]

    async def load_db():
        for card in mongo.get_cards():
            resp = card
            resp["id"] = "load_db";
            # await asyncio.sleep(0.1) # slight delay to avoid sending packets together
            asyncio.create_task(
                send_message(
                    json.dumps(resp)
                )
            )

    async def new_card(data):
        del data['id']
        mongo.add_card(data)

    async def translate_a(data):
        print(data)
        msg = translate_text(data['msg'], target_language=data['targetLocale'])

        asyncio.create_task(
            send_message(
                json.dumps({
                    'msg': msg,
                    'id': 'translate_ans',
                    'targetLocale': data['targetLocale'],
                    'index': data['index']
                })
            )
        )

    def check_single_tag(string):
        opening_tag = "<s>"
        closing_tag = "<e>"
        
        count = 0
        
        # Iterate over the string to find tags
        i = 0
        while i < len(string):
            # Check if the current part of the string is an opening tag
            if string[i:i+len(opening_tag)] == opening_tag:
                count += 1
                i += len(opening_tag)
            # Check if the current part of the string is a closing tag
            elif string[i:i+len(closing_tag)] == closing_tag:
                count -= 1
                if count < 0:
                    # More closing tags than opening tags
                    return False
                i += len(closing_tag)
            else:
                # Move to the next character if no tag is found
                i += 1
        
        # If count is zero, all opening tags have matching closing tags
        return count == 0

    def split_by_tag(string):
        # Construct the regex pattern for the tag
        pattern = rf"<s>(.*?)<e>"
        
        # Find all matches using the regex pattern
        segments = re.findall(pattern, string, re.DOTALL)
        
        return segments

    async def receive_messages():
        global compiled_t
        while True:
            raw = await reader.read(100)

            # add to compiled
            compiled_t = compiled_t + raw.decode()

            if not raw:
                break

            if check_single_tag(compiled_t):
                for data in split_by_tag(compiled_t):
                    message = data
                    print(f"Received: {message}")

                    data_json = json.loads(message)
                    if data_json["id"] == "load_db":
                        print("Loading from database!")
                        asyncio.create_task(load_db())

                    elif data_json["id"] == "translate_title":
                        print("Recieved translation request!")
                        asyncio.create_task(translate_title(data_json))

                    elif data_json["id"] == "new_card":
                        print("Received new card request!")
                        asyncio.create_task(new_card(data_json))

                    elif data_json["id"] == "translate_ans":
                        print("Received answer translation request")
                        asyncio.create_task(translate_a(data_json))

                compiled_t = ""


            # Optionally, process the message and send a response
            """
            response = "Message received".encode()
            writer.write(response)
            await writer.drain()
            print("Response sent")
            """

    async def send_messages():
        while True:
            # Example: sending a periodic message
            await asyncio.sleep(5)  # Wait for 5 seconds
            message = "Periodic message from server".encode()
            writer.write(message)
            await writer.drain()
            print("Periodic message sent")

    async def send_message(message):
        message = '<s>' + message + '<e>' # add markers
        writer.write(message.encode())
        await writer.drain()
        print("Sent", message)


    try:
        # await asyncio.gather(receive_messages(), send_messages())
        await asyncio.gather(receive_messages())
    except asyncio.CancelledError:
        pass
    finally:
        print(f"Closing connection with {addr}")
        writer.close()

        try:
            await writer.wait_closed()
        except Exception as e:
            print(e)

async def main():

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip = s.getsockname()[0]
    print(ip)
    s.close()

    server = await asyncio.start_server(handle_client, ip, 65432)
    addr = server.sockets[0].getsockname()
    print(f'Serving on {addr}')

    async with server:
        await server.serve_forever()

if __name__ == '__main__':
    asyncio.run(main())
