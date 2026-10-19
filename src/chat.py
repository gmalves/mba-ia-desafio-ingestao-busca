from search import search_prompt

EXIT_COMMANDS = {"sair", "exit", "quit"}


def main():
    chain = search_prompt()

    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return

    print("Chat iniciado. Faça sua pergunta (digite 'sair' para encerrar).\n")

    while True:
        try:
            question = input("Faça sua pergunta:\n\nPERGUNTA: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEncerrando o chat...")
            break

        if not question:
            continue

        if question.lower() in EXIT_COMMANDS:
            print("Encerrando o chat...")
            break

        answer = chain.invoke(question)
        print(f"\nRESPOSTA: {answer}\n\n---\n")


if __name__ == "__main__":
    main()
