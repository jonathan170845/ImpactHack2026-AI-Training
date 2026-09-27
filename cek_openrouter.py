"""Kirim satu pesan uji ke OpenRouter menggunakan konfigurasi .env proyek."""

import os
from pathlib import Path
import sys

import requests
from dotenv import load_dotenv


ENV_PATH = Path(__file__).resolve().parent / ".env"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
ERROR_HINTS = {
    "400": "Periksa ID model dan parameter permintaan.",
    "401": "Periksa OPENROUTER_API_KEY; key mungkin salah atau sudah dinonaktifkan.",
    "402": "Periksa saldo akun serta batas kredit API key di OpenRouter.",
    "403": "Periksa izin model dan pembatasan akun atau API key.",
    "404": "Periksa OPENROUTER_MODEL dan ketersediaannya di katalog OpenRouter.",
    "429": "Batas permintaan tercapai. Tunggu sebelum mencoba lagi.",
    "502": "Provider model sedang bermasalah. Coba lagi nanti.",
    "503": "Provider model tidak tersedia. Coba lagi nanti atau pilih model lain.",
}


def main() -> int:
    # Gunakan isi terbaru file lokal meskipun terminal sudah memuat .env sebelumnya.
    load_dotenv(ENV_PATH, override=True)
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    model = os.getenv("OPENROUTER_MODEL", "").strip()
    if not api_key or not model:
        print(
            "Isi OPENROUTER_API_KEY dan OPENROUTER_MODEL di file .env "
            "yang berada di folder yang sama dengan skrip ini.",
            file=sys.stderr,
        )
        return 1

    print("Mengirim satu pesan uji ke OpenRouter...", flush=True)
    try:
        response = requests.post(
            API_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [
                    {"role": "user", "content": "Balas singkat: Halo dari ReviewLens."}
                ],
                "max_tokens": 256,
                "stream": False,
            },
            timeout=(10, 60),
        )
    except requests.Timeout:
        print("Permintaan ke OpenRouter melewati batas waktu. Coba lagi nanti.", file=sys.stderr)
        return 1
    except requests.RequestException:
        print("Gagal menghubungi OpenRouter. Periksa koneksi internet atau proxy.", file=sys.stderr)
        return 1

    try:
        payload = response.json()
    except ValueError:
        print(f"Respons OpenRouter bukan JSON (HTTP {response.status_code}).", file=sys.stderr)
        return 1

    error = payload.get("error") if isinstance(payload, dict) else None
    if not response.ok or error is not None:
        code = str(response.status_code)
        if isinstance(error, dict):
            code = str(error.get("code", response.status_code))
        print(f"Permintaan OpenRouter gagal (kode {code}).", file=sys.stderr)
        if code in ERROR_HINTS:
            print(ERROR_HINTS[code], file=sys.stderr)
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            print(error["message"].replace(api_key, "[API key disembunyikan]"), file=sys.stderr)
        return 1

    try:
        choice = payload["choices"][0]
        content = choice["message"]["content"]
    except (KeyError, IndexError, TypeError):
        print("Respons OpenRouter tidak memuat balasan model yang valid.", file=sys.stderr)
        return 1

    if choice.get("finish_reason") == "error":
        print("Provider mengalami error saat menghasilkan balasan. Coba lagi nanti.", file=sys.stderr)
        return 1
    if not isinstance(content, str) or not content.strip():
        print("OpenRouter merespons, tetapi balasan teks model kosong.", file=sys.stderr)
        if choice.get("finish_reason") == "length":
            print(
                "Batas token habis sebelum balasan teks tersedia. Naikkan max_tokens "
                "di skrip atau pilih model tanpa reasoning untuk uji sederhana ini.",
                file=sys.stderr,
            )
        return 1

    print("Koneksi OpenRouter berhasil.")
    print(f"Balasan model: {content.strip()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
