# Reporting and Fixing Security Issues

Please do not open GitHub issues or pull requests - this makes the problem immediately visible to everyone, including malicious actors.

For security issues in GeoFeatureKit, please reach out directly at **lihangalex@pm.me** as opening issues or pull requests may expose vulnerabilities to potential attackers.

## What to Include

When reporting a security issue, please include:

- A clear description of the vulnerability
- Steps to reproduce the issue
- Potential impact assessment
- Any suggested fixes or mitigations

## Supported Versions

We provide security updates for:

| Version | Supported |
|---------|-----------|
| 0.6.x   | ✅ Yes    |
| 0.5.x   | ❌ No     |
| < 0.5   | ❌ No     |

## Security Best Practices

When using GeoFeatureKit:

- **Keep dependencies updated**: Regularly update to the latest version
- **Validate inputs**: Sanitize coordinates and parameters from untrusted sources  
- **Rate limiting**: Be mindful of API rate limits when using external data sources
- **Cache security**: Ensure cached data doesn't contain sensitive information

---

Thank you for helping keep GeoFeatureKit secure! 🔒 