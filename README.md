# Construction Waste Recovery — Vercel Deployment

Static HTML frontend (`public/index.html`) + Python serverless function
(`api/analyze.py`) wrapping the same 4-agent LangGraph pipeline used in
the Streamlit version.

## Deploy

1. Push this folder to a GitHub repo (or a fresh repo, separate from the
   Streamlit one — the two are structured differently).
2. Go to vercel.com → "Add New Project" → import the repo.
3. Before deploying, add an Environment Variable:
   - Key: `GROQ_API_KEY`
   - Value: your key from console.groq.com
4. Deploy. Vercel auto-detects `public/` as static output and
   `api/analyze.py` as a Python serverless function.
5. Visit your deployed URL — the page in `public/index.html` calls
   `/api/analyze` automatically.

## Notes

- `vercel.json` sets `maxDuration: 60` for the function, since the
  pipeline makes 3 sequential Groq calls. If your plan doesn't allow
  60s functions, this may need adjusting downward — Groq is fast, but
  don't add extra agents without checking the timeout budget.
- `api/test_value_agent.py` still covers the deterministic value
  calculator — run locally with `pytest api/test_value_agent.py`.
