const diffai = require('../index.js');

describe('diff()', () => {
    describe('Basic API', () => {
        test('diff function exists', () => {
            expect(typeof diffai.diff).toBe('function');
        });

        test('diffPaths function exists', () => {
            expect(typeof diffai.diffPaths).toBe('function');
        });

        test('formatOutput function exists', () => {
            expect(typeof diffai.formatOutput).toBe('function');
        });
    });

    describe('Basic Diff Operations', () => {
        test('returns empty array for identical objects', () => {
            const obj = { a: 1, b: 2 };
            const results = diffai.diff(obj, obj);
            expect(Array.isArray(results)).toBe(true);
            expect(results.length).toBe(0);
        });

        test('detects added keys', () => {
            const old = { a: 1 };
            const newObj = { a: 1, b: 2 };
            const results = diffai.diff(old, newObj);
            expect(results.length).toBeGreaterThanOrEqual(1);
            const addedResult = results.find(r => r.diffType === 'Added' && r.path === 'b');
            expect(addedResult).toBeDefined();
        });

        test('detects removed keys', () => {
            const old = { a: 1, b: 2 };
            const newObj = { a: 1 };
            const results = diffai.diff(old, newObj);
            expect(results.length).toBeGreaterThanOrEqual(1);
            const removedResult = results.find(r => r.diffType === 'Removed' && r.path === 'b');
            expect(removedResult).toBeDefined();
        });

        test('detects modified values', () => {
            const old = { a: 1 };
            const newObj = { a: 2 };
            const results = diffai.diff(old, newObj);
            expect(results.length).toBe(1);
            expect(results[0].diffType).toBe('Modified');
            expect(results[0].path).toBe('a');
            expect(results[0].oldValue).toBe(1);
            expect(results[0].newValue).toBe(2);
        });
    });

    describe('ML/AI Specific Features', () => {
        test('handles tensor-like data', () => {
            const old = {
                layers: [
                    { weight: [1.0, 2.0, 3.0] }
                ]
            };
            const newObj = {
                layers: [
                    { weight: [1.0, 2.0, 4.0] }
                ]
            };
            const results = diffai.diff(old, newObj);
            expect(Array.isArray(results)).toBe(true);
        });

        test('handles model metadata', () => {
            const old = {
                model: 'resnet50',
                epoch: 10,
                loss: 0.5
            };
            const newObj = {
                model: 'resnet50',
                epoch: 20,
                loss: 0.3
            };
            const results = diffai.diff(old, newObj);
            expect(results.length).toBeGreaterThan(0);
        });
    });

    describe('Options', () => {
        test('accepts epsilon option', () => {
            const old = { value: 1.0 };
            const newObj = { value: 1.0001 };

            const resultsWithoutEpsilon = diffai.diff(old, newObj);
            const resultsWithEpsilon = diffai.diff(old, newObj, { epsilon: 0.001 });

            expect(resultsWithoutEpsilon.length).toBe(1);
            expect(resultsWithEpsilon.length).toBe(0);
        });

        test('accepts pathFilter option', () => {
            const old = { a: 1, b: 2 };
            const newObj = { a: 2, b: 3 };
            const results = diffai.diff(old, newObj, { pathFilter: 'a' });
            expect(results.length).toBe(1);
            expect(results[0].path).toBe('a');
        });
    });

    describe('Format Output', () => {
        test('formats results as JSON', () => {
            const old = { a: 1 };
            const newObj = { a: 2 };
            const results = diffai.diff(old, newObj);
            const formatted = diffai.formatOutput(results, 'json');
            expect(typeof formatted).toBe('string');
            expect(() => JSON.parse(formatted)).not.toThrow();
        });

        test('formats results as diffai format', () => {
            const old = { a: 1 };
            const newObj = { a: 2 };
            const results = diffai.diff(old, newObj);
            const formatted = diffai.formatOutput(results, 'diffai');
            expect(typeof formatted).toBe('string');
        });
    });

    describe('Error Handling', () => {
        test('handles empty objects', () => {
            const results = diffai.diff({}, {});
            expect(Array.isArray(results)).toBe(true);
            expect(results.length).toBe(0);
        });

        test('handles nested objects', () => {
            const old = { nested: { deep: { value: 1 } } };
            const newObj = { nested: { deep: { value: 2 } } };
            const results = diffai.diff(old, newObj);
            expect(results.length).toBe(1);
        });
    });
});
