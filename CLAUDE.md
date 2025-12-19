# diffai-js の思想（Philosophy）

diffai-js は diffai-core の Node.js バインディングです。
AI/MLモデルファイル（PyTorch, Safetensors, NumPy, MATLAB）の比較機能を提供します。

## 構造

```
diffai-js/
├── src/lib.rs      # Rust NAPI バインディング
├── index.js        # Node.js エントリポイント
├── Cargo.toml      # Rust 依存関係
├── package.json    # npm パッケージ設定
└── tests/          # Jest テスト
```

## 開発

```bash
npm install
npm run build
npm test
```

## API

- `diff(old, new, options?)` - 2つのオブジェクトを比較
- `diffPaths(oldPath, newPath, options?)` - 2つのファイル/ディレクトリを比較
- `formatOutput(results, format)` - 結果をフォーマット

## 重要

- diffai-core は crates.io から参照
- バージョンは diffai-core と同期
