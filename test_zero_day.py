from zero_day_defense import ZeroDayDefense

defense = ZeroDayDefense()

test_urls = [
    'https://google.com',
    'https://github.com',
    'http://paypal-login-security.xyz',
    'http://verify-account-paypal.com',
    'https://docs.google.com/document/d/abc123',
    'https://api.stripe.com/v1/customers',
    'https://myapp.vercel.app',
    'https://example.com/blog/post/123',
    'https://shop.example.com/product?id=123',
]

print('=' * 70)
print('ZERO-DAY DEFENSE - QUICK SCAN TEST')
print('=' * 70)

for url in test_urls:
    result = defense.analyze(url)
    print(f'\n{url}')
    print(f'  Verdict: {result["final_verdict"]}')
    print(f'  Risk: {result["risk_score"]} | Confidence: {result["confidence"]}%')
    print(f'  Reasons: {result["layers"]["heuristics"]["reasons"][:3]}')
    if result['layers']['brand_monitor']['typosquatting']:
        print(f'  Typosquatting: {result["layers"]["brand_monitor"]["matched_brand"]}')