"""Collect up to 1000 Facebook group comments; save only the content column."""
import argparse
import csv
import re
import shutil
import time
from pathlib import Path

from playwright.sync_api import TimeoutError as PlaywrightTimeout, sync_playwright

GROUP_IDS = ("408571091380061", "584397217391365", "450310163844833")
POST_URL = re.compile(r"/groups/(\d+)/(?:posts|permalink)/(\d+)")
BLOCKED = re.compile(r"temporarily blocked|tạm thời bị chặn|captcha|confirm your identity|xác nhận danh tính", re.I)
MORE = r"more comments|previous comments|view all \d+ comments|more replies|xem thêm bình luận|xem các bình luận trước|xem thêm phản hồi|xem thêm câu trả lời|xem tất cả \d+ bình luận"

# Take leaf text inside comment articles, excluding profile links and UI controls.
EXTRACT = """() => [...document.querySelectorAll('[role="article"]')]
  .filter(a => /comment|bình luận/i.test(a.getAttribute('aria-label') || ''))
  .flatMap(a => [...a.querySelectorAll('[dir="auto"]')]
    .filter(e => !e.querySelector('[dir="auto"], a, button, [role="button"]'))
    .filter(e => !e.closest('a, button, [role="button"]'))
    .map(e => e.innerText.trim()))
  .filter(s => s && !/^(Like|Reply|Thích|Trả lời|See more|Xem thêm|Admin|Author|Top contributor)$/i.test(s))"""
LINKS = "() => [...document.querySelectorAll('a[href]')].map(a => a.href)"
CLICK_MORE = """pattern => {
  const re = new RegExp(pattern, 'i');
  const button = [...document.querySelectorAll('a,button,[role="button"]')]
    .find(e => e.getClientRects().length && re.test(e.innerText || ''));
  if (button) button.click();
  return !!button;
}"""


def load_seen(path):
    if not path.exists():
        return set()
    with path.open(encoding="utf-8", newline="") as f:
        return {row["content"] for row in csv.DictReader(f) if row.get("content")}


def save(page, path, seen, target):
    fresh = []
    for text in page.evaluate(EXTRACT):
        text = " ".join(text.split())
        if text and text not in seen:
            seen.add(text)
            fresh.append(text)
        if len(seen) >= target:
            break
    if fresh:
        with path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(["content"])
            writer.writerows((text,) for text in fresh)
    return len(fresh)


def accessible(page):
    body = page.locator("body").inner_text(timeout=5000)[:5000]
    if BLOCKED.search(body):
        raise RuntimeError("Facebook yêu cầu xác minh hoặc báo chặn; dừng.")
    if page.locator('input[name="email"]').count() and "Get more from Groups" in body:
        raise RuntimeError("Phiên chưa đăng nhập; dừng.")


def collect_group(page, group_id, path, seen, args):
    url = f"https://www.facebook.com/groups/{group_id}/"
    page.goto(url, wait_until="domcontentloaded", timeout=30000)
    posts, stale = set(), 0
    for _ in range(args.max_feed_scrolls):
        accessible(page)
        added = save(page, path, seen, args.target)
        old_post_count = len(posts)
        for link in page.evaluate(LINKS):
            match = POST_URL.search(link)
            if match and match.group(1) == group_id:
                posts.add(f"https://www.facebook.com/groups/{group_id}/posts/{match.group(2)}/")
        print(f"{group_id}: {len(seen)}/{args.target} bình luận, {len(posts)} bài viết", flush=True)
        if len(seen) >= args.target or len(posts) >= args.max_posts:
            break
        stale = stale + 1 if added == 0 and len(posts) == old_post_count else 0
        if stale >= 12:
            break
        page.mouse.wheel(0, 1800)
        time.sleep(args.pause)

    for post in list(posts)[:args.max_posts]:
        if len(seen) >= args.target:
            break
        try:
            page.goto(post, wait_until="domcontentloaded", timeout=30000)
            stale_comments = 0
            for _ in range(args.max_comment_clicks):
                accessible(page)
                added = save(page, path, seen, args.target)
                if len(seen) >= args.target:
                    break
                clicked = page.evaluate(CLICK_MORE, MORE)
                if not clicked:
                    page.mouse.wheel(0, 1200)
                stale_comments = stale_comments + 1 if added == 0 and not clicked else 0
                if stale_comments >= 3:
                    break
                time.sleep(args.pause)
            print(f"{len(seen)}/{args.target} bình luận", flush=True)
        except PlaywrightTimeout:
            print(f"Bỏ qua bài tải quá lâu: {post}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=Path("data/comments.csv"))
    parser.add_argument("--max-feed-scrolls", type=int, default=300)
    parser.add_argument("--max-posts", type=int, default=500)
    parser.add_argument("--max-comment-clicks", type=int, default=50)
    parser.add_argument("--pause", type=float, default=1.5)
    parser.add_argument("--no-login", action="store_true", help="Try the limited public view")
    args = parser.parse_args()
    seen = load_seen(args.output)
    if len(seen) >= args.target:
        print(f"Đã có {len(seen)} bình luận trong {args.output}")
        return
    with sync_playwright() as p:
        browser = p.chromium.launch_persistent_context(
            str(Path(".fb-profile").resolve()), headless=args.no_login,
            executable_path=shutil.which("google-chrome")
        )
        page = browser.new_page()
        page.set_default_timeout(5000)
        try:
            if not args.no_login:
                page.goto("https://www.facebook.com/", wait_until="domcontentloaded")
                input("Đăng nhập trong cửa sổ Facebook, rồi nhấn Enter ở terminal: ")
            for group_id in GROUP_IDS:
                if len(seen) >= args.target:
                    break
                collect_group(page, group_id, args.output, seen, args)
        except RuntimeError as error:
            print(error)
        finally:
            browser.close()
    print(f"Hoàn tất: {len(seen)}/{args.target} bình luận trong {args.output}")


if __name__ == "__main__":
    main()
