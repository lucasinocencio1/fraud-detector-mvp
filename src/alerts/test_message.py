#sandbox to test the message sending

import time
import random

def send_fake_message(phone, message):
    """Simula o envio de mensagem sem API real."""
    print(f"\n--- Enviando mensagem para {phone} ---")
    time.sleep(random.uniform(0.5, 1.5))  # simula latência de rede
    print(f"Mensagem: {message}")
    print("✅ Mensagem enviada (simulação)\n")


if __name__ == "__main__":
    # Simula 3 mensagens para números fictícios
    fake_numbers = ["+351910000001", "+351920000002", "+351930000003"]
    for num in fake_numbers:
        msg = (
            f"⚠️ Alerta: possível fraude detectada!\n"
            f"Cliente fictício {num[-3:]}\n"
            f"Valor: €{random.randint(200, 5000)}\n"
            f"Status: em revisão automática."
        )
        send_fake_message(num, msg)
