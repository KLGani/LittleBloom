# Backend setup

1. Copy `.env.example` to `.env.local`. This file is ignored by Git.
2. In MongoDB Atlas, create a database user, allow your server's IP address in
   Network Access, and copy the application's connection string into `MONGO_URI`.
   Replace the username, password, and cluster host in the example URI. Do not
   commit the real connection string.
3. Set `JWT_SECRET` to a long, random value and keep it unchanged between server
   restarts. A changed secret invalidates existing login tokens.
4. From the repository root, start the screening service in its own terminal:

   ```powershell
   cd ml_service
   python -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   .\.venv\Scripts\python.exe ml_server.py
   ```

   The trained `pregnancy_model.pkl` and `child_model.pkl` files must be in
   `ml_service`. To retrain them, run `python train.py` from that directory.
5. In the backend environment, set `ML_SERVICE_URL` if the Python service is not
   running on the same machine (`http://127.0.0.1:5001` by default). For a
   separately hosted service, bind it to a reachable interface with `ML_HOST`
   and point `ML_SERVICE_URL` at that host.
6. In a separate terminal, change to `backend` and start the API with
   `npm install` followed by `npm run dev`.

Users and password hashes are stored in the MongoDB database selected by
`MONGO_URI`, so they remain available after the API server restarts. The
frontend uses the active Expo host (or the browser's host) by default. Set
`EXPO_PUBLIC_API_URL` only when the API is hosted at a different address; use a
LAN-reachable address for a physical device.

The classifier uses the supplied demo datasets and is not clinically validated.
Its predictions are exercised for integration purposes only and do not trigger
health alerts. The app uses its configured vital-sign thresholds for screening;
these are not a diagnosis. The model runs only when all its inputs are
available. Missing inputs or an unavailable ML service are reported rather
than being replaced with guessed measurements.
