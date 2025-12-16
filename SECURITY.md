# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.x     | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it by emailing the maintainers directly. Do not open a public GitHub issue.

**Response Timeline:**
- Initial response: within 48 hours
- Status update: within 7 days
- Resolution target: within 30 days (severity dependent)

## Security Considerations

This application:
- Does not store user credentials
- Does not process payment information
- Uses read-only access to public F1 timing data via FastF1 API

### Data Handling

- Session data is cached locally for performance
- No personal data is collected or transmitted
- All visualizations are generated from publicly available race data

### Dependencies

We regularly update dependencies to address known vulnerabilities. Run `pip list --outdated` to check for updates.
