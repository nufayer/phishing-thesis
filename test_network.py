from feature_extractor import extract_features

# Test with include_network=False (default)
f1 = extract_features('https://docs.google.com/document/d/abc123', include_network=False)
print('Without network:')
print('  DomainAgeDays:', f1.get('DomainAgeDays'))
print('  SSLCertValid:', f1.get('SSLCertValid'))
print('  HasSPFRecord:', f1.get('HasSPFRecord'))
print('  HasDMARCRecord:', f1.get('HasDMARCRecord'))

# Test with include_network=True (default might be True)
f2 = extract_features('https://docs.google.com/document/d/abc123', include_network=True)
print('With network:')
print('  DomainAgeDays:', f2.get('DomainAgeDays'))
print('  SSLCertValid:', f2.get('SSLCertValid'))
print('  HasSPFRecord:', f2.get('HasSPFRecord'))
print('  HasDMARCRecord:', f2.get('HasDMARCRecord'))