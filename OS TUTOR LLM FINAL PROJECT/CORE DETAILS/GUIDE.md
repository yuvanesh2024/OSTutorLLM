# OSTutorLLM: full implementation guide

## 1. What your uploaded submissions actually require

Review 1 proposes a course-aware OS tutor: use an open language model, retrieve institutional notes, explain theory, create labs/viva/MCQs, adapt practice, and keep deployment affordable. It explicitly makes instruction fine-tuning optional. Review 2 documents a retrieval-only Colab experiment: one thread/concurrency slide deck, 2,630 extracted characters, two chunks, MiniLM embeddings and an in-memory Chroma database. Qwen was not connected. The deadlock question did not have an answer in that source, so the returned neighbours do not establish retrieval accuracy. The reported 0.20 seconds is retrieval time for that tiny experiment, not complete answer latency.

This implementation starts fresh. Do not reuse old Colab variables or assume those earlier results demonstrate this new build. The two uploaded reports describe the project; use actual OS teaching notes as its knowledge base.

## 2. What we are building and why

A real, working LLM application has three main jobs here:

1. Retrieve relevant course text using embeddings plus keyword ranking.
2. Ask an actual pretrained language model to explain that evidence.
3. Present an answer with inspectable page/slide references and practice tools.

The unique addition is an interactive CPU scheduling verification lab. Python computes exact FCFS, non-preemptive SJF, and Round Robin results; the interface draws a Gantt chart and displays completion, turnaround and waiting times. The tutor can explain the concepts alongside the verified numbers. This makes a useful OS demonstration without another model, agent framework, or large training dataset.

This is a pretrained Qwen3-based RAG tutor. It is not a foundation model pretrained by the team, and it is not fine-tuned. Present it honestly as **“OSTutorLLM: a course-grounded OS tutor with adaptive practice and a CPU scheduling verification lab.”** If faculty separately requires changed model weights, obtain that requirement explicitly; RAG alone does not meet a mandatory fine-tuning requirement.

## 3. Recommended platform and stack

| Item | Selection | Reason |
|---|---|---|
| Development and execution | Google Colab, GPU runtime | No local CUDA installation needed |
| Reference GPU | T4-class 16 GiB GPU | Target for this small quantized model; availability varies |
| Persistent project files | Your own mounted Google Drive | Retain notes, extracted chunks, progress and results |
| Answering model | `Qwen/Qwen3-4B`, 4-bit NF4 | More manageable than 14B for a student prototype |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2`, CPU | Fast, small, suitable for a first working build |
| Retrieval | Chroma cosine search + BM25 rank fusion | Semantic retrieval plus exact-term ranking |
| Browser UI | Gradio | Avoid a separate frontend/backend deployment |
| Progress | SQLite | Simple saved per-topic attempts |
| Scheduling lab | Pure Python + Matplotlib | Deterministic calculations and plots |

The 14B plan in Review 1 is not necessary for the first demonstration. Four-bit 14B weights alone are roughly 7 GB before quantization overhead, runtime buffers, and attention cache; do not assume an 8 GB GPU comfortably fits the whole system. For this guide use 4B first. If memory is tight, change the notebook model ID to `Qwen/Qwen3-1.7B`, restart and rerun. Smaller models may produce weaker explanations and MCQs.

You need a browser, a Google account, internet for setup/model downloads, GPU runtime access, and storage for model downloads. Budget at least 15 GB free temporary disk plus space for course material. The cloud workflow is not an offline demo: Colab and its public tunnel need internet. A fully local setup is a separate deployment exercise.

## 4. Files in this package

- `START_HERE.ipynb`: complete, self-contained Colab notebook with explanations and executable code in phase order. Start here.
- `core.py`: extraction, chunking, embeddings/retrieval, prompts, MCQ persistence/adaptation, evaluation, scheduling calculations.
- `ui.py`: three-tab interface.
- `requirements.txt`: reference dependency versions.
- `tests/test_core.py`: deterministic backend regression tests.
- `data/course/DEMO_NOTES.md`: original scheduling notes for a tiny smoke test only.
- `GUIDE.md`: this guide, including troubleshooting and review instructions.

The notebook contains full copies of the code and writes them into `/content/ostutor_app`; you do not need to upload each Python file. The notebook also includes the demo notes and tests. Uploading only `START_HERE.ipynb` to Colab is sufficient to begin.

## 5. The complete build sequence

### Phase 1 — open a fresh Colab notebook and enable GPU

1. Extract the ZIP on your computer.
2. Open https://colab.research.google.com/ in a browser.
3. Click File → Upload notebook. Select `START_HERE.ipynb`.
4. Click Runtime → Change runtime type → select a GPU such as T4 → Save.
5. Click Connect. Execute cells with Shift+Enter in the order shown.
6. Run the first check cell. `CUDA available: True` must appear before Phase 7.

**What this means:** Colab is the computer running your Python program. Your laptop is displaying and controlling it. The browser is not doing the model calculations. GPU availability and session duration are not guaranteed on the free service.

### Phase 2 — install the software

1. Run the requirements-writing cell.
2. Run `%pip install -q -r /content/ostutor-requirements.txt`.
3. Wait for completion. Warnings are not always failures; inspect any red exception or unresolved dependency error.
4. If asked to restart the session, do so, then continue from Phase 3.

The pinned direct dependencies are a reference setup, not a fully tested lockfile for every future Colab image. Do not casually upgrade one component and expect the rest to remain compatible. Keep the actual `pip freeze` output from the successful runtime.

**Checkpoint:** imports in the later backend and model cells should work. Installing software does not download or train the LLM yet.

### Phase 3 — mount Drive and create folders

1. Run the Drive cell and authorize your own account.
2. It creates `MyDrive/OSTutorLLM/data/course` for teaching files and `MyDrive/OSTutorLLM/results` for evidence of evaluation.
3. It creates `/content/ostutor_app` on the temporary runtime for program code.
4. Note the printed paths. These are Colab Linux paths, not Windows folders.

Saved file locations:

| File/folder | Purpose | Survives complete runtime reset? |
|---|---|---|
| `MyDrive/OSTutorLLM/data/course` | Source teaching documents | Yes |
| `MyDrive/OSTutorLLM/chunks.json` | Extracted text with source identifiers | Yes |
| `MyDrive/OSTutorLLM/progress.sqlite3` | Questions, answer keys, student attempts | Yes |
| `MyDrive/OSTutorLLM/results` | Test reports and environment records | Yes |
| `/content/ostutor_chroma` | Active disk-backed vector index | No; rebuild from Drive files |
| `/content/ostutor_app` | Runtime code | No; notebook rewrites it |
| Downloaded model cache on Colab | Checkpoints | Usually no; may download again |

Do not run two sessions against the same progress file simultaneously. Student names are pseudonyms, not secure accounts. This is a controlled course prototype.

### Phase 4 — write the implementation

Run the two `%%writefile` cells. Each cell creates a full Python module in `/content/ostutor_app`.

Do not paste the code into the Windows terminal or Google Drive's search bar. These are notebook code cells. The `%`/`%%` commands are notebook commands, not ordinary Python syntax for `.py` files.

The included implementation intentionally avoids LangChain/LlamaIndex because direct library calls make this small pipeline easier to explain and debug. RAG does not require either framework.

### Phase 5 — gather and upload teaching material

1. Begin with one readable CPU scheduling deck or PDF.
2. Add notes on processes and threads, synchronization, deadlocks, paging/memory management, and a lab manual.
3. Later add file systems, shell notes, and previous question papers if available and permitted.
4. Prefer descriptive filenames, for example `CPU_Scheduling.pdf` and `Deadlocks.pptx`.
5. Execute `files.upload()` and select those files. The cell copies them to the project course folder in Drive.
6. Do not put the project proposal/progress reports into the knowledge base as if they were OS lessons.
7. Avoid putting unrelated documents in the course folder; everything with a supported extension is indexed recursively.

For the very first smoke test you may run the optional demo-notes cell instead. It supports scheduling questions only. Remove `DEMO_NOTES.md` from Drive before a formal faculty-material evaluation.

**Scanned material:** if text cannot be selected in a PDF, extraction may return nothing. Image-only slide diagrams are also unreadable by this parser. Use a permitted OCR tool or write a checked `.txt` transcript and upload it. A source with no chunks is not successfully indexed. Poor OCR can silently introduce mistakes; inspect it manually.

### Phase 6 — extract, chunk, embed, index, and inspect

1. Run the ingestion/index cell.
2. Read the extraction table: filename, characters, chunks, status.
3. Investigate every `ERROR` or `NO TEXT` entry. Do not proceed while your central topic is unreadable.
4. Inspect the printed first chunk. It should contain meaningful course content, with the correct source and one-based page/slide number.
5. Run the retrieval test using a question that appears in those notes.
6. Open the original cited page/slide and verify that it answers the question.

The extractor keeps pages/slides separate. Long units become 140-word windows with 25 words of overlap. This is a small, understandable starting configuration; MiniLM has a short input limit, so inspect long technical text and shorten chunks if token truncation affects retrieval. It reads PDF text, slide text, tables, grouped text shapes and available speaker notes; not embedded image text.

Embeddings map each chunk to numbers so semantically similar questions can find it. Chroma searches these vectors. BM25 also ranks keyword overlap. Reciprocal-rank fusion combines the rankings. The same embedding model is used for documents and questions.

The initial cosine cutoff is **0.32**, a tuning starting point. It is not “32% confidence.” It may admit irrelevant content or reject relevant content. Phase 10 measures this. Even when a question passes the cutoff, the model can still answer incorrectly; citations and manual checks remain necessary.

Edited files: upload the replacement with the same name, rerun ingestion and construct a new Retriever. Removed files: delete them from the Drive course folder and rebuild. Content-based collection names keep old indexed content from being selected for the new corpus. Old cache collections may remain on temporary disk but are not searched.

### Phase 7 — load Qwen and verify plain generation

1. Run the model-loading cell.
2. Wait for checkpoint download, quantization and GPU loading to finish.
3. The cell loads `Qwen/Qwen3-4B` using Hugging Face Transformers and bitsandbytes 4-bit NF4.
4. It defines the `generate()` function and tests a one-sentence OS response.
5. Confirm a real model answer appears before proceeding.

MiniLM is the search embedding model. Qwen is the language-generating LLM. They perform different jobs. The code sets `enable_thinking=False`, uses a bounded answer length, and serializes generation. It uses greedy generation for reproducibility, so quiz wording may repeat for the same topic and difficulty.

No Hugging Face token is ordinarily required for these public model IDs. A token warning alone is not a failed model load. Access or network errors must be resolved before the application can work.

### Phase 8 — connect the tutor and practice system

1. Create `Tutor(retriever, generate, ...)` using the notebook cell.
2. Ask a supported question in “Explain simply” mode.
3. Read the answer and its evidence excerpts. Verify its source labels.
4. Try “Exam answer,” “Viva practice,” and “Shell lab” through the UI later.
5. Generate one MCQ on a covered topic. Answer it and inspect the explanation.
6. Use a fresh pseudonym for the real demo, so the notebook's test attempt does not affect the demonstration history.

The system gives the LLM source text plus a prompt asking it to use that evidence. If no source passes the retrieval threshold, generation is skipped and it asks for relevant notes. The backend checks that numeric citation labels fall within the supplied source range; it cannot prove entailment or stop every hallucination.

MCQs must have four distinct options, a valid answer index, an explanation and valid source IDs. Invalid JSON is retried once, then reported. Formatting checks do not prove that an MCQ has exactly one substantively correct answer; manually review sample questions before presenting them as valid assessment content.

**Adaptive rule:** fewer than three attempts → easy. After that, use up to the latest five answers for the same student/topic: below 50% → easy; 50% to below 80% → medium; at least 80% → hard. The LLM is instructed to generate at that level. Difficulty is not psychometrically calibrated. Duplicate submissions for one quiz are blocked.

Shell scripts are displayed for learning. The application never executes generated code. Test any code separately in a disposable lab folder before using it in a submission.

### Phase 9 — launch the interface and test the unique feature

1. Run the UI launch cell.
2. Enter a demo username and a password of at least eight characters.
3. Open the displayed Gradio URL and sign in.
4. Use the “Ask your notes” tab for a self-contained OS question.
5. Expand the source evidence section and check the actual page/slide.
6. Use “Adaptive practice” to generate, answer and grade MCQs; view saved progress.
7. Open “Verify CPU scheduling.” Keep the sample input and compare algorithms.
8. Change one burst or arrival, calculate again, and explain what changed.

The temporary Gradio tunnel exposes the running demo outside your machine. Protect it with the chosen password and permitted teaching material. It stops when Colab disconnects. This is a shared demonstration password, not separate secure student accounts. Pseudonyms do not prevent someone with the demo password from entering another pseudonym.

**Expected numerical results:**

Input:
```text
P1,0,5
P2,1,3
P3,2,1
```

| Algorithm | Completion P1/P2/P3 | Waiting P1/P2/P3 | Average waiting |
|---|---|---|---|
| FCFS | 5 / 8 / 9 | 0 / 4 / 6 | 3.33 |
| SJF, non-preemptive | 5 / 9 / 6 | 0 / 5 / 3 | 2.67 |
| Round Robin, quantum 2 | 9 / 8 / 5 | 4 / 4 / 2 | 3.33 |

Turnaround = completion − arrival. Waiting = turnaround − burst. Tie-breaking uses arrival then input order; SJF first compares burst length among ready processes. In Round Robin, arrivals at the end of a time slice enter the ready queue before the interrupted process is requeued.

The calculator assumes a single CPU, one CPU burst per process, no I/O, and zero context-switch overhead. It validates IDs, positive bursts, nonnegative arrivals, input size and quantum. These assumptions must be stated during the review. The scheduling explanation button generates a conceptual explanation grounded in notes and appends the exact Python results. It does not make the LLM responsible for the arithmetic.

### Phase 10 — evaluate and keep evidence

1. Create 30–50 questions and manually identify the supporting page/slide for every answerable question.
2. Include questions from topics intentionally absent from the uploaded material and questions unrelated to OS.
3. Split into development and held-out test cases before tuning.
4. Write `evaluation.json` in the project Drive root using the notebook's exact schema.
5. Use the development subset to choose the similarity cutoff. Compare supported-question evidence hits and negative-question abstention; neither alone is sufficient.
6. Set `THRESHOLD` once based on development results and run the test cell.
7. Inspect the saved CSV. Record corpus size, embedding model, LLM, threshold, device and answer lengths.
8. Separately ask the tutor the test questions and manually judge answer correctness, citation support and appropriate abstention. Record full answer latency including generation, not just vector search.
9. Review a sample of generated MCQs and shell exercises for factual and executable correctness.
10. Run `python -m unittest discover -s tests -v` in the notebook. These tests check backend behaviour, not LLM quality.

The automatic metric is **Evidence Hit@4**: for an answerable question, did any expected source page/slide appear in the top four retrieved chunks? It is not a complete multi-source recall measure or an answer-accuracy percentage. Negative abstention measures how often no evidence passes for an unsupported question. Manually inspect false positive/negative retrieval.

No benchmark scores are claimed in this package. Eleven backend tests passed during preparation, and notebook Python cells were syntax-checked. CUDA model inference, actual retrieval-model downloads, Chroma integration and the live Gradio UI need your Colab execution; they were not run end-to-end in the preparation environment.

### Phase 11 — save, rehearse and submit

1. Save a copy of the notebook to Drive.
2. Run the reproducibility-record cell to save package versions and run metadata.
3. Capture screenshots of your own successful extraction, sourced answer, quiz feedback, progress and scheduling chart.
4. Record a short backup video of the working model.
5. Keep the actual notebook, permitted data, code, evaluation JSON/CSV and screenshots ready for the faculty.
6. Rehearse the explanation below and the six-minute demo sequence.

**Restart checklist:** choose GPU → install → mount Drive → write backend/UI modules → skip upload unless files changed → rebuild from saved course files → load Qwen → recreate Tutor → launch UI. Old quiz attempts persist in Drive. The model and temporary index may need downloading/rebuilding again.

## 6. Suggested implementation schedule

| Session | Work | Done when |
|---|---|---|
| 1 | Setup, one real deck, parsing and retrieval | Relevant source page/slide found and verified |
| 2 | Load Qwen, connect RAG, add missing-topic response | Supported question answered with inspectable evidence |
| 3 | Browser UI and scheduling demo | All three algorithms match the expected example |
| 4 | Add remaining material, quizzes and progress checks | Attempts survive recreating Tutor/restarting |
| 5 | Evaluate, fix failures, rehearse | Evidence files and backup recording are ready |

These are work sessions, not guaranteed calendar durations. Model downloads, GPU availability and OCR can change the time required. If you have fewer sessions, prioritize a complete sourced Q&A path and scheduling demo before adding more documents or polish.

## 7. A faculty explanation in simple words

“We use an existing open language model, Qwen3, rather than training a foundation model from scratch. We extract text from OS teaching files, split it into page-aware chunks and create semantic embeddings. For a question, semantic and keyword retrieval find relevant evidence. Qwen explains that evidence and gives source references. We also generate practice questions and adapt their requested difficulty using saved topic-wise performance. Our extra feature computes CPU schedules directly in Python, so completion and waiting times can be verified independently of the LLM.”

Suggested six-minute demonstration:

1. Show the actual course files and extraction report.
2. Ask “What is a race condition?” only if the loaded material covers it; open the supporting slide.
3. Ask a question from a deliberately missing topic and show the result honestly. If the heuristic fails to abstain, report it as a limitation; do not stage a fake success.
4. Generate and answer a question; show the saved progress/difficulty rule.
5. Compare FCFS, SJF and RR with the same input, then change an input live.
6. Show the evaluation report and explain what the metrics measure.

Do not claim you trained Qwen, completely eliminated hallucinations, achieved a research result, or implemented offline hosting. Fine-tuning, learned personalization, production identity management and deployment beyond Colab are future work here.

## 8. Troubleshooting

| Symptom | Action |
|---|---|
| `CUDA available: False` | Select a GPU runtime. If no quota is available, retry later or use an accessible GPU runtime. |
| CUDA out of memory | Restart to clear old loaded models. Use Qwen3-1.7B, smaller output limits or fewer retrieved chunks. Do not keep multiple model instances. |
| `KeyError: qwen3` | Confirm Transformers 4.51+; install the provided requirements and restart if necessary. |
| Download/network/rate-limit failure | Keep internet connected, inspect the error, retry the failed download later. A warning without an exception may be harmless. |
| No text/chunks | Check PDF text selection. OCR or transcribe image-only sources and inspect the result. |
| “Not enough support” for a real topic | Check extraction, question phrasing, and retrieved scores. Tune on development data; do not simply remove the threshold. |
| Irrelevant material retrieved | Check corpus coverage, compare negative examples, tune the threshold, reduce noisy text and improve your documents. |
| Wrong answer despite plausible sources | Read excerpts, refine prompts/source content and record it in evaluation. RAG is not a guarantee. |
| Invalid quiz twice | Generate again. Inspect real model output if repeated. Reduce topic ambiguity or use a stronger model when feasible. |
| Same quiz wording repeats | Greedy decoding favours reproducibility. Repetition is a limitation; do not market this as an unlimited question bank. |
| SQLite lock/Drive I/O issue | Stop other notebooks using the same folder. Use one controlled session and check Drive connectivity. |
| Gradio URL stops | Reconnect Colab and rerun launch; a new URL may be issued. |
| Follow-up “why?” gives poor results | Each request is independent. Ask a self-contained question naming the topic. |
| New file absent in results | Rebuild the index and recreate Tutor after uploading; old Retriever objects do not update themselves. |

## 9. Official technical references

- Qwen3-4B model card, model usage and thinking switch: https://huggingface.co/Qwen/Qwen3-4B
- Transformers 4.56.2 bitsandbytes integration: https://huggingface.co/docs/transformers/v4.56.2/quantization/bitsandbytes
- Chroma collection configuration/cosine distance: https://docs.trychroma.com/docs/collections/configure
- Gradio Blocks and launch/authentication: https://www.gradio.app/docs/gradio/blocks

Use these references to explain the tools; your own test run supplies evidence that your application works.
