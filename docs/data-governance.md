# Data governance

DEON 6 training data must have documented provenance and a usable license or explicit permission.

Before a dataset enters training:
1. Record its source and license.
2. Record acquisition date and dataset statistics.
3. Run language identification.
4. Remove personal data where required.
5. Deduplicate exact and near duplicates.
6. Run quality filtering.
7. Check contamination against evaluation sets.
8. Keep the manifest with the experiment record.

Never commit private or restricted data to the repository.
