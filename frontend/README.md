# Cough Capture: Front End

## Run it

Needs [Node.js](https://nodejs.org/) (LTS).

```bash
cd frontend
npm install
npm run dev
```

Open the address it prints (usually http://localhost:5173).

## Setup

Create a `.env` file next to `package.json`:

```
VITE_GOOGLE_CLIENT_ID=your-id.apps.googleusercontent.com
```

Ask Rich for the ID. Restart `npm run dev` after editing it.

## Packages

`npm install` installs everything: `react`, `react-dom`, `vite`, `@vitejs/plugin-react`.

