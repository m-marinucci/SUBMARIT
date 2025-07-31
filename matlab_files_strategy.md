# Strategy for Original MATLAB Files

## Recommended Approach: Create a Separate Directory

### Option 1: Archive in the Repository (RECOMMENDED)
Move all MATLAB files to a dedicated directory to maintain them for reference while keeping the Python code clean.

```bash
# Create archive directory
mkdir -p matlab_original

# Move all .m files
mv *.m matlab_original/

# Create a README for the MATLAB files
```

### Option 2: Create a Separate Branch
Keep MATLAB files in a separate branch for historical reference.

```bash
# Create and switch to matlab-original branch
git checkout -b matlab-original

# Commit all MATLAB files
git add *.m
git commit -m "Archive original MATLAB implementation"

# Switch back to main and remove MATLAB files
git checkout main
git rm *.m
git commit -m "Remove MATLAB files from main branch (archived in matlab-original branch)"
```

### Option 3: Create a Separate Repository
For larger projects, maintain MATLAB code in a separate repository.

## Why Keep the MATLAB Files?

1. **Reference Implementation**: Useful for verifying numerical accuracy
2. **Documentation**: MATLAB comments may contain implementation details
3. **Validation**: Can run both versions to compare outputs
4. **Legacy Users**: Some users may still need MATLAB version
5. **Historical Record**: Preserves the original implementation

## Recommended Directory Structure

```
SubmarketIdentificationTesting/
├── src/submarit/          # Python implementation
├── tests/                 # Python tests
├── docs/                  # Documentation
├── examples/              # Python examples
├── matlab_original/       # Original MATLAB files
│   ├── README.md         # Explanation of MATLAB files
│   ├── CreateSubstitutionMatrix.m
│   ├── kSMLocalSearch.m
│   └── ... (other .m files)
└── README.md             # Main project README
```

## Additional Considerations

### 1. Add MATLAB README
Create a README in the matlab_original directory explaining:
- These are the original MATLAB implementations
- The Python version is the actively maintained version
- How to use MATLAB files for validation
- Link to migration guide

### 2. Update .gitignore
Ensure .gitignore excludes MATLAB temporary files:
```
*.m~
*.asv
*.mat
```

### 3. License Considerations
- Ensure MATLAB code license is compatible
- Document any licensing differences

### 4. CI/CD Updates
- Exclude MATLAB files from Python linting/testing
- Optionally add MATLAB validation tests if MATLAB is available

## Decision Matrix

| Approach | Pros | Cons | Best For |
|----------|------|------|----------|
| Archive in repo | Easy access, single repo | Larger repo size | Most projects |
| Separate branch | Clean main branch | Switching branches needed | Git-savvy teams |
| Separate repo | Complete separation | More complex management | Large projects |
| Delete files | Cleanest structure | Loss of reference | Not recommended |

## Recommendation

**Archive the MATLAB files in a `matlab_original/` directory** with a clear README. This approach:
- Keeps files accessible for reference
- Maintains clean separation from Python code
- Preserves historical implementation
- Supports validation and comparison
- Is simple to implement and understand