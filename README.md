# RAG με LangChain, Chroma και Ollama

Το πρόγραμμα αναζητά αποσπάσματα σε τοπικά PDF, TXT και Markdown αρχεία και απαντά με παραπομπές. Χρησιμοποιεί το embeddinggemma για embeddings και το gemma3:12b για απαντήσεις μέσω Ollama.

## Εκτέλεση

Από τον φάκελο του repository, με ενεργό το Python virtual environment:

    python -m pip install -r requirements.txt
    python rag.py index document.pdf notes.txt article.md
    python rag.py ask "What does the document say?"
    python rag.py ask

Η εντολή index προσθέτει ή ενημερώνει αρχεία. Αν ένα αρχείο δεν έχει αλλάξει, δεν υπολογίζει ξανά τα embeddings του. Η εντολή ask χρησιμοποιεί τη βάση που έχει αποθηκευτεί, χωρίς να ξαναδιαβάζει τα αρχικά αρχεία. Το Ollama πρέπει να τρέχει και τα μοντέλα embeddinggemma και gemma3:12b να είναι διαθέσιμα.

## Πώς λειτουργεί

## Πώς λειτουργεί

1. Οι loaders διαβάζουν PDF ανά σελίδα ή TXT/Markdown ως κείμενο.
2. Το chunking.py χωρίζει το κείμενο σε αποσπάσματα.
3. Το vector_store.py αποθηκεύει τα αποσπάσματα και τα embeddings στο τοπικό Chroma, μέσα στο data/chroma.
4. Το retrieval.py επιλέγει έως TOP_K αποσπάσματα για κάθε ερώτηση.
5. Τα citations.py και generation.py μορφοποιούν τις πηγές και παράγουν την απάντηση.
6. Το rag.py παρέχει τις εντολές index και ask.

PDF χωρίς εξαγώγιμο κείμενο χρειάζονται OCR πριν από την εισαγωγή τους.

Το conflict_prompt.py ζητά από το μοντέλο να αναφέρει αντιφάσεις. Δεν αποτελεί ακόμη αυτόνομο μηχανισμό εντοπισμού ή επίλυσης conflicts.
