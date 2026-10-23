# OSTutorLLM

**A course-grounded Operating Systems tutor with adaptive practice and a CPU scheduling verification lab.**

OSTutorLLM connects a pretrained Qwen3 language model to uploaded OS teaching material through Retrieval-Augmented Generation (RAG). Students can inspect the retrieved evidence, practise questions, track topic-wise performance, and calculate CPU schedules through a browser interface.

This project was developed for an Operating Systems course submission. It integrates existing pretrained models; it does not pretrain or fine-tune a foundation model.

## Features

- **Course-grounded answers:** retrieve passages from PDF, PPTX, TXT and Markdown teaching files.
- **Inspectable evidence:** display source filenames, page/slide numbers and retrieved excerpts.
- **Four answer styles:** simple explanations, exam answers, viva practice and shell-lab exercises.
- **Adaptive MCQs:** generate four-option questions, grade submissions and request difficulty using recent topic-wise performance.
- **Saved progress:** store quiz records and attempts in SQLite; reject duplicate submissions.
- **Scheduling verification:** calculate FCFS, non-preemptive SJF and Round Robin schedules with tables and Gantt charts.
- **Deterministic arithmetic:** Python computes scheduling results; the language model provides conceptual explanations.

## Technology stack

| Component | Implementation |
|---|---|
| Answering model | `Qwen/Qwen3-4B` |
| Inference | Hugging Face Transformers and bitsandbytes, 4-bit NF4 |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2`, running on CPU |
| Vector index | Chroma with cosine distance |
| Keyword ranking | BM25 |
| Rank combination | Reciprocal rank fusion with a semantic eligibility threshold |
| Interface | Gradio |
| Progress storage | SQLite |
| Document extraction | pypdf and python-pptx |
| Tables and plots | pandas and Matplotlib |
| Reference environment | Google Colab with a CUDA GPU and mounted Google Drive |

## How it works

1. Extract text from the teaching documents while retaining source locations.
2. Split pages/slides into overlapping text chunks.
3. Encode chunks with MiniLM and store them in Chroma.
4. Retrieve semantic candidates and combine their ranks with BM25 results.
5. Pass eligible evidence and the question to Qwen.
6. Display the generated answer and the evidence used to produce it.

The default search returns up to four chunks after ranking. The initial cosine-similarity threshold is `0.32`; it is a configurable heuristic, **not a confidence percentage**. BM25 reranks candidates within the eligible dense shortlist; lexical-only matches outside that shortlist are not returned.

If no evidence passes the threshold, the tutor skips generation and asks for relevant material. Prompt instructions and citation checks reduce some failure modes but do not guarantee factual correctness.

## Repository files

| File | Purpose |
|---|---|
| [`START_HERE.ipynb`](START_HERE.ipynb) | Self-contained Colab setup, model loading, application launch and regression-test cells |
| [`core.py`](core.py) | Ingestion, retrieval, tutor, quiz storage, adaptation and scheduling algorithms |
| [`ui.py`](ui.py) | Three-tab Gradio interface |
| [`requirements.txt`](requirements.txt) | Reference Python dependencies |
| [`GUIDE.md`](GUIDE.md) | Detailed implementation and troubleshooting guide |
| [`.gitignore`](.gitignore) | Excludes local data, databases, model weights, caches and credentials |

Keep the notebook and standalone modules consistent when making changes: the notebook contains embedded copies of the implementation.

## Run in Google Colab

### 1. Open the notebook and select a GPU

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload `START_HERE.ipynb`, or open the notebook from this repository through Colab.
3. Select **Runtime → Change runtime type → GPU**.
4. Run the CUDA check cell. Model loading requires `torch.cuda.is_available()` to be `True`.

A T4-class GPU with approximately 16 GiB VRAM is the reference target. GPU availability and session duration vary. The first model download needs internet and several GB of disk space; allow at least 15 GB of free temporary storage.

### 2. Install dependencies

Run the notebook's requirements-writing and installation cells. Keep Colab's CUDA-enabled PyTorch rather than replacing it blindly.

The supplied requirements describe a reference environment, not a guarantee of compatibility with every future Colab image. Restart the session if requested after installation. If installation reports conflicts, inspect the named packages and verify project imports before proceeding.

### 3. Mount Drive and add course documents

Run the Drive setup cell. It creates this course-material directory:

```text
/content/drive/MyDrive/OSTutorLLM/data/course
```

Upload permitted OS notes, slides and lab manuals into **My Drive → OSTutorLLM → data → course**.

You can either upload directly through Drive or use the notebook's upload cell. Do not do both for the same files. Project proposals and progress reports are not substitutes for OS teaching content.

Image-only files need OCR or checked transcripts. Diagrams are not interpreted by the text extractor. The optional demo notes in the notebook support only a small scheduling smoke test.

### 4. Build and inspect the index

Run the backend/UI writing cells and then the ingestion/indexing cell. Check that:

- The extraction report lists your documents.
- Important files yield meaningful text and nonzero chunks.
- The indexed count matches the extracted chunk count.
- A question covered by your notes retrieves relevant source material.

### 5. Load Qwen and connect the tutor

Run the Qwen model-loading cell, wait for its test answer, and then run the Tutor setup cell. Compare a course-grounded answer with the displayed source excerpts.

If GPU memory is insufficient, restart and use the documented smaller-model fallback, `Qwen/Qwen3-1.7B`. Record the actual model used when reporting results.

### 6. Launch the interface

Run the Gradio launch cell and choose a demo username and password. Open the displayed public URL.

| Tab | Use |
|---|---|
| **Ask your notes** | Submit a self-contained OS question and inspect source evidence |
| **Adaptive practice** | Generate an MCQ, choose an answer, submit and view progress |
| **Verify CPU scheduling** | Enter processes, select an algorithm and calculate a chart |

The public share link connects to the active Colab backend. It is temporary and stops working when the runtime stops. The current setup is not independently hosted.

## CPU scheduling example

Enter one process per line, without a header:

```text
P1,0,5
P2,1,3
P3,2,1
```

Each row is `PID,arrival,burst`. For Round Robin, use quantum `2`.

| Algorithm | Completion P1 / P2 / P3 | Waiting P1 / P2 / P3 | Average waiting | Average turnaround |
|---|---|---|---:|---:|
| FCFS | 5 / 8 / 9 | 0 / 4 / 6 | 3.33 | 6.33 |
| Non-preemptive SJF | 5 / 9 / 6 | 0 / 5 / 3 | 2.67 | 5.67 |
| Round Robin, quantum 2 | 9 / 8 / 5 | 4 / 4 / 2 | 3.33 | 6.33 |

**Assumptions:** one CPU, one CPU burst per process, no I/O blocking and zero context-switch cost. Arrivals at a Round Robin slice boundary enter the queue before the running process is requeued. Arrival ties follow input order; SJF first compares ready-process burst lengths.

A small local check can run without loading either model. From the repository root:

```bash
python -c "from core import schedule; print(schedule('P1,0,5\nP2,1,3\nP3,2,1', 'FCFS')[0])"
```

This checks the numerical backend only. Running `python ui.py` does not start the complete application; use the notebook to construct the retriever, generator and Tutor first.

## Adaptive practice

Requested difficulty is calculated per student pseudonym and topic:

| Recent attempts | Next requested difficulty |
|---|---|
| Fewer than 3 | Easy |
| At least 3, recent accuracy below 50% | Easy |
| At least 3, recent accuracy 50% to below 80% | Medium |
| At least 3, recent accuracy at least 80% | Hard |

The calculation uses up to the latest five attempts. Progress summaries show all-time totals, so their overall accuracy can differ from the window controlling difficulty. This is a rule-based heuristic; generated question difficulty is not independently calibrated.

## Validation

The notebook includes 11 backend regression tests covering scheduling, input validation, ingestion, quiz structure, persistence, duplicate grading and abstention control flow. Run its test-folder, test-writing and unittest cells in order. After those cells create `tests/test_core.py`, the notebook runs:

```bash
python -m unittest discover -s tests -v
```

Backend tests use a fake generator where appropriate. They do not establish real-model accuracy, retrieval quality, GPU compatibility or browser availability.

For formal retrieval evaluation, the notebook supports a separate `evaluation.json` with questions and manually verified supporting pages. This is optional for running the app. Do not claim accuracy or latency figures without recording the corresponding experiment.

## Storage and restarts

| Artifact | Location | Recovery |
|---|---|---|
| Course files and `chunks.json` | Google Drive project directory | Retained across runtime resets |
| `progress.sqlite3` | Google Drive project directory | Reopen using the same path |
| Active Chroma index | Colab temporary storage | Rebuild from saved teaching files |
| Loaded model and runtime modules | Colab runtime | Reload/recreate after reset |
| Environment and run metadata | Project `results` directory | Save through the final notebook cell |

Use one active notebook writer per project database. Saving the notebook does not preserve a running GPU session. After a full reset, reinstall dependencies, mount Drive, write the modules, rebuild the index, load Qwen, recreate Tutor and relaunch the UI. Upload documents again only if they changed.

## Limitations

- Retrieval depends on readable, relevant teaching material and threshold selection.
- Generated answers and MCQs can be incorrect even when their structure is valid.
- Source-ID checks verify citation range, not factual entailment.
- Each question is independent; conversation history is not retained.
- Greedy generation may repeat quiz wording.
- Student pseudonyms are not individual authenticated accounts.
- The Colab/Gradio setup requires internet and an active runtime.
- Generated shell scripts are displayed only and require separate review/testing.

## Publishing this repository

Commit the notebook, source code, requirements and documentation. Keep teaching material private unless redistribution is permitted. Do not commit student databases, model downloads or credentials. The supplied `.gitignore` covers common generated artifacts but cannot remove secrets typed into source or notebook outputs.

Before uploading a notebook, use **Edit → Clear all outputs** on a publication copy and inspect its cells for tokens, passwords and private material. Keep your executed evidence copy separately for the academic submission. `.gitignore` does not affect files already tracked by Git.

## Future work

- Labelled retrieval evaluation and answer-support review.
- OCR and token-aware chunking.
- Question deduplication and faculty review tools.
- Individual authentication and independent deployment.
- Additional OS calculators, such as page replacement or SRTF.

## References

- [Retrieval-Augmented Generation — Lewis et al.](https://arxiv.org/abs/2005.11401)
- [Sentence-BERT — Reimers and Gurevych](https://aclanthology.org/D19-1410/)
- [Qwen3-4B model card](https://huggingface.co/Qwen/Qwen3-4B)
- [MiniLM embedding model](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [Chroma documentation](https://docs.trychroma.com/)
- [Gradio documentation](https://www.gradio.app/docs)

## License

No project license is specified by this README. Add a `LICENSE` file if you choose to grant reuse rights. Model weights and teaching documents remain subject to their own licenses and permissions.
