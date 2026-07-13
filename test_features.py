from feature_extractor import brand_similarity, extract_features

test_domains = [
    'docs.google.com',
    'api.stripe.com', 
    'shop.example.com',
    'example.com',
    'drive.google.com',
    'mail.google.com',
    'calendar.google.com',
    'api.github.com',
    'dashboard.stripe.com',
    'google.com',
    'stripe.com',
    'github.com',
    'paypal-login-security.xyz',
    'verify-account-paypal.com',
]

print('Brand Similarity Test:')
for d in test_domains:
    sim = brand_similarity(d)
    print(f'  {d:<35} -> {sim:.3f}')

print()
print('Full feature test:')
urls = [
    'https://docs.google.com/document/d/abc123',
    'https://api.stripe.com/v1/customers',
    'https://shop.example.com/product?id=123',
    'https://example.com/blog/post/123',
    'https://docs.google.com/document/d/abc123/edit',
]

for url in urls:
    f = extract_features(url)
    print(url)
    bs = f.get('BrandSimilarity', 0)
    print(f'  BrandSimilarity: {bs:.3f}, PathLength: {f["PathLength"]}, URLEntropy: {f["URLEntropy"]:.3f}')
    print()