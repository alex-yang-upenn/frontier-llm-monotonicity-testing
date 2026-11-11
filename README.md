# Testing LLMs for Monotonicity

We ask an LLM to act as a hiring screener: it reads a real job posting and a candidate's skills section, and answers **accept** or **reject**.

A sensible screener should be monotone: adding a skill to a resume should never turn an accept into a reject.

To check this, we treat the LLM as a boolean function over skills:

- The resume always lists a fixed set of baseline skills.
- On top of that, there are N optional "stretch" skills, each either included (1) or left out (0).
- So every resume is an N-bit string, and the LLM is a function `f: {0,1}^N → {accept, reject}`.

2^N (N = 35 in this codebase) unique resumes is on the order of ~34 billion LLM calls. We use property testing techniques to minimize resource usage. A key constraint is that each LLM call costs real money, so we also track tokens and dollars required to perform the monotonicity tests.

We implement the Edge Tester from Goldreich et al. *Testing Monotonicity* (2000) and the Distance Estimator from Fattal & Ron, *Approximating the Distance to Monotonicity in High Dimensions* (2010).

---

## Quickstart

```bash
# Create a virtual environment and install dependencies
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt

# Create a file named `.env` in the project root, and fill it with your Azure OpenAI credentials:
echo "AZURE_API_KEY=your-key" >> .env
echo "AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/openai/v1" >> .env
echo "AZURE_PROJECT_ENDPOINT=https://<your-project-endpoint>" >> .env

# Run the program
python run_experiment.py  # calls the LLM (costs money), prints and saves results to `results/run_<timestamp>.json`
```

**Command-line options**

| Option | What it does | Default |
|---|---|---|
| `--dimension` | Dimension n of the boolean hypercube, i.e. the number of stretch skills (takes the first n, most relevant first) | all 35 |
| `--deployment-name` | Azure OpenAI deployment to query | `gpt-6.1-sol` |

**Code layout**

| File | What it holds |
|---|---|
| `run_experiment.py` | Command-line entry point; wires everything together |
| `monotonicity_testing/edge_tester.py` | Goldreich et al. edge tester and its result |
| `monotonicity_testing/fattal_ron_estimator.py` | Fattal & Ron distance estimator and its result |
| `monotonicity_testing/exact_distance.py` | Exact distance of a subcube restriction (used by Fattal–Ron) |
| `monotonicity_testing/screening_client.py` | The LLM screener (Azure OpenAI, prompt caching; remembers answers so no resume is paid for twice) |
| `monotonicity_testing/token_pricing.py` | Token counts and dollar cost |
| `monotonicity_testing/skills_catalog.py` | Turns a bit string into a resume skills section |
| `data/` | The job posting, the skills list, and where both came from |


---



## References

- O. Goldreich, S. Goldwasser, E. Lehman, D. Ron, A. Samorodnitsky. *Testing Monotonicity.* Combinatorica, 2000.
- S. Fattal, D. Ron. *Approximating the Distance to Monotonicity in High Dimensions.* ACM Transactions on Algorithms, 2010.
