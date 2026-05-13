import os
import re
import base64
import json
import email
import html
import pandas as pd
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

CREDENTIALS_FILE = "credentials.json"   # path to OAuth credentials file
TOKEN_FILE       = "token.json"         
MAX_EMAILS       = None
OUTPUT_EMAILS    = "emails_dataset.csv"
OUTPUT_URLS      = "urls_dataset.csv"

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

# URL regex pattern
URL_PATTERN = re.compile(
    r'https?://[^\s\'"<>)\]]+',
    re.IGNORECASE
)

def authenticate():
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, "w") as f:
            f.write(creds.to_json())
    return creds


def decode_part(data: str) -> str:
    try:
        return base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="replace")
    except Exception:
        return ""


def extract_body(payload: dict) -> str:
    mime_type = payload.get("mimeType", "")
    body_data = payload.get("body", {}).get("data", "")

    if mime_type == "text/plain" and body_data:
        return decode_part(body_data)

    if mime_type == "text/html" and body_data:
        raw_html = decode_part(body_data)
        # Strip HTML tags for a clean plain-text body
        clean = re.sub(r"<[^>]+>", " ", raw_html)
        clean = html.unescape(clean)
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean

    
    parts = payload.get("parts", [])
    for part in parts:
        result = extract_body(part)
        if result:
            return result

    return ""


def get_header(headers: list, name: str) -> str:
    for h in headers:
        if h["name"].lower() == name.lower():
            return h["value"]
    return ""


def parse_sender_name(from_header: str) -> str:
    match = re.match(r'^"?([^"<]+?)"?\s*<', from_header)
    if match:
        return match.group(1).strip()
    return from_header.strip()


# ─────────────────────────────────────────────
# MAIN EXTRACTION
# ─────────────────────────────────────────────
def fetch_emails(service, max_emails):
    messages = []
    page_token = None

    print("Fetching message IDs from inbox...")
    while True:
        params = {"userId": "me", "labelIds": ["INBOX"], "maxResults": 500}
        if page_token:
            params["pageToken"] = page_token

        response = service.users().messages().list(**params).execute()
        batch = response.get("messages", [])
        messages.extend(batch)

        print(f"  Retrieved {len(messages)} message IDs so far...")

        if max_emails and len(messages) >= max_emails:
            messages = messages[:max_emails]
            break

        page_token = response.get("nextPageToken")
        if not page_token:
            break

    print(f"Total messages to process: {len(messages)}")
    return messages


def process_messages(service, message_ids):
    email_rows = []
    url_rows   = []
    seen_urls  = set()

    for i, msg_ref in enumerate(message_ids, 1):
        if i % 50 == 0:
            print(f"  Processing message {i}/{len(message_ids)}...")

        try:
            msg = service.users().messages().get(
                userId="me",
                id=msg_ref["id"],
                format="full"
            ).execute()
        except Exception as e:
            print(f"  Warning: could not fetch message {msg_ref['id']}: {e}")
            continue

        headers     = msg.get("payload", {}).get("headers", [])
        sender_raw  = get_header(headers, "From")
        sender_name = parse_sender_name(sender_raw)
        subject     = get_header(headers, "Subject")
        body        = extract_body(msg.get("payload", {}))

        email_rows.append({
            "sender_name": sender_name,
            "subject":     subject,
            "body":        body,
            "label":       0
        })


        combined_text = f"{subject} {body}"
        urls = URL_PATTERN.findall(combined_text)

        for url in urls:
            # Clean trailing punctuation that's not part of the URL
            url = url.rstrip(".,;:!?)\"'")
            if url not in seen_urls:
                seen_urls.add(url)
                url_rows.append({"url": url, "label": 0})

    return email_rows, url_rows



def main():
    if not os.path.exists(CREDENTIALS_FILE):
        print(
            f"\n[ERROR] '{CREDENTIALS_FILE}' not found.\n"
            "Please follow the setup instructions at the top of this script\n"
            "to create OAuth credentials from Google Cloud Console.\n"
        )
        return

    print("Authenticating with Gmail...")
    creds   = authenticate()
    service = build("gmail", "v1", credentials=creds)

    message_ids = fetch_emails(service, MAX_EMAILS)

    print("\nProcessing messages...")
    email_rows, url_rows = process_messages(service, message_ids)

    # ── Save emails dataset ────────────────────
    df_emails = pd.DataFrame(email_rows, columns=["sender_name", "subject", "body", "label"])
    df_emails.to_csv(OUTPUT_EMAILS, index=False, encoding="utf-8-sig")
    print(f"\n✓ Saved emails dataset  →  {OUTPUT_EMAILS}  ({len(df_emails)} rows)")

    # ── Save URLs dataset ──────────────────────
    df_urls = pd.DataFrame(url_rows, columns=["url", "label"])
    df_urls.to_csv(OUTPUT_URLS, index=False, encoding="utf-8-sig")
    print(f"✓ Saved URLs dataset    →  {OUTPUT_URLS}  ({len(df_urls)} unique URLs)")

    print("\nDone!")


if __name__ == "__main__":
    main()
