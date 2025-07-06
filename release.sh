#!/bin/bash

# GeoFeatureKit Release Script
# Usage: ./release.sh 0.2.5

set -e

if [ $# -eq 0 ]; then
    echo "Usage: $0 <version>"
    echo "Example: $0 0.2.5"
    exit 1
fi

NEW_VERSION=$1
echo "🚀 Starting release process for version $NEW_VERSION"

# Validate version format
if [[ ! $NEW_VERSION =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "❌ Invalid version format. Use semantic versioning (e.g., 0.2.5)"
    exit 1
fi

# Check if we're on main branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$CURRENT_BRANCH" != "main" ]; then
    echo "❌ Must be on main branch to release. Current branch: $CURRENT_BRANCH"
    exit 1
fi

# Check if working directory is clean
if [ -n "$(git status --porcelain)" ]; then
    echo "❌ Working directory is not clean. Please commit or stash changes."
    exit 1
fi

# Update version in setup.py
echo "📝 Updating version in setup.py..."
sed -i.bak "s/version=\"[^\"]*\"/version=\"$NEW_VERSION\"/" setup.py
rm setup.py.bak

# Verify the change
UPDATED_VERSION=$(python -c "import re; content=open('setup.py').read(); print(re.search(r'version=\"([^\"]+)\"', content).group(1))")
if [ "$UPDATED_VERSION" != "$NEW_VERSION" ]; then
    echo "❌ Failed to update version in setup.py"
    exit 1
fi

echo "✅ Version updated to $NEW_VERSION"

# Run tests to make sure everything works
echo "🧪 Running tests..."
python -m pytest -x

echo "✅ All tests passed"

# Commit version bump
echo "📝 Committing version bump..."
git add setup.py
git commit -m "Bump version to $NEW_VERSION"

# Push to main
echo "📤 Pushing to main branch..."
git push origin main

echo "🎉 Version bump completed!"
echo "✅ Version $NEW_VERSION has been committed and pushed to main"
echo ""
echo "🚀 Next steps to publish:"
echo "   1. Go to: https://github.com/lihangalex/geofeaturekit/actions"
echo "   2. Click on 'Publish to PyPI' workflow"
echo "   3. Click 'Run workflow' button"
echo "   4. Enter version: $NEW_VERSION"
echo "   5. Click 'Run workflow' to publish"
echo ""
echo "🔄 The workflow will then automatically:"
echo "   - Verify version matches setup.py"
echo "   - Build the package"
echo "   - Publish to PyPI"
echo "   - Create git tag"
echo "   - Create GitHub release"
echo ""
echo "📦 Package will be available at: https://pypi.org/project/geofeaturekit/$NEW_VERSION/" 