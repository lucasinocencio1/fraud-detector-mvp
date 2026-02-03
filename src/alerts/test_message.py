# #sandbox to test the message sending
# import time
# import random

# def send_fake_message(phone, message):
#     """Simula o envio de mensagem sem API real."""
#     print(f"\n--- Sending message to {phone} ---")
#     time.sleep(random.uniform(0.5, 1.5))  # simulate network latency
#     print(f"Message: {message}")
#     print("Message sent (simulation)\n")


# if __name__ == "__main__":
#     # Simulate 3 messages to fake numbers
#     fake_numbers = ["+351910000001", "+351920000002", "+351930000003"]
#     for num in fake_numbers:
#         msg = (
#             "Fraud alert triggered.\n"
#             f"Customer stub {num[-3:]}\n"
#             f"Amount: €{random.randint(200, 5000)}\n"
#             "Status: under automated review."
#         )
#         send_fake_message(num, msg)
