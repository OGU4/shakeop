# shakeop — 設計ドキュメント

> Splatoon 3「サーモンラン NEXT WAVE」リアルタイム解析オペレーターアプリ
> 
> このドキュメントはClaude Codeでの開発ガイドとして使用する。

---

## 目次

1. [プロジェクト概要](#1-プロジェクト概要)
2. [技術スタックと確定事項](#2-技術スタックと確定事項)
3. [アーキテクチャ全体像](#3-アーキテクチャ全体像)
4. [映像入力モジュール](#4-映像入力モジュール)
5. [認識パイプライン詳細](#5-認識パイプライン詳細)
6. [ゲーム状態管理 (FSM)](#6-ゲーム状態管理-fsm)
7. [音声通知システム](#7-音声通知システム)
8. [GUIデザイン仕様](#8-guiデザイン仕様)
9. [設定・データモデル](#9-設定データモデル)
10. [ディレクトリ構成](#10-ディレクトリ構成)
11. [開発フェーズとタスク](#11-開発フェーズとタスク)
12. [モデル学習ガイド](#12-モデル学習ガイド)
13. [クロスプラットフォーム対応メモ](#13-クロスプラットフォーム対応メモ)

---

## 1. プロジェクト概要

### 何を作るか

サーモンラン NEXT WAVEのゲーム画面をリアルタイムに解析し、以下の情報をプレイヤーに音声＋GUIで通知するデスクトップアプリケーション。

- Wave番号と潮位（通常/満潮/干潮）
- 金イクラのノルマと現在数
- 残り時間
- オオモノシャケの出現と種類
- 特殊Wave（ラッシュ、グリル、ハコビヤ、霧、ドロシャケ噴出等）の検知
- リザルト（クリア/失敗、最終スコア）

### なぜ作り直すか

既存の ShakeScouter-NW + Shake-Streamkit-NW（GPL-3.0 fork）には以下の問題がある:

1. 2プロセス起動が必要 → **1アプリ・2〜3クリックで稼働**にしたい
2. テンプレートマッチングが環境差（明るさ等）に弱い → **ML+pHash+特徴量の多段パイプライン**で解決
3. ブラウザベースGUIの音声再生制約 → **ネイティブ音声再生**で解決
4. GPL-3.0 → **MITライセンス**でクリーンルーム実装

### ターゲットユーザー

- サーモンランのカンスト（でんせつ999）を目指すプレイヤー
- 配信者（OBS連携でオーバーレイ表示したい人）
- オペレーター役を自動化したいチーム

---

## 2. 技術スタックと確定事項

| 項目 | 選定 | 備考 |
|---|---|---|
| 言語 | Python 3.11+ | CV/ML処理とGUIを統一 |
| 環境管理 | uv | pyproject.toml ベース、lockfile対応 |
| GUI | PySide6 (Qt 6) | LGPL、ネイティブ音声再生可能 |
| 映像入力 | OpenCV (VideoCapture) | OBS仮想カメラ対応。NDIは将来対応 |
| シーン分類 | ONNX Runtime + YOLOv8-cls | 軽量分類モデル |
| オブジェクト検出 | ONNX Runtime + YOLOv8n/s | オオモノシャケ検出 |
| 数字認識 | pHash + カスタム数字分類器 | ROI切り出し→1文字分類。汎用OCR不使用 |
| 固定テキスト認識 | パーセプチュアルハッシュ (pHash) | 環境差に強い画像指紋比較 |
| 特徴量マッチング | OpenCV ORB/AKAZE | UIアイコン補助検出 |
| 音声再生 | QSoundEffect (Qt Multimedia) | .wav事前ロード、低遅延 |
| 音声生成 | VOICEVOX (事前生成・.wav同梱) | オフライン動作、キャラ音声 |
| 配布 | PyInstaller or Nuitka | シングルバイナリ |
| OS | Windows 10/11 + Linux (Ubuntu 22.04+) | |
| ライセンス | MIT | |

---

## 3. アーキテクチャ全体像

```
┌──────────────────────────────────────────────────────────────┐
│                        shakeop                           │
│                                                              │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────────┐  │
│  │   GUI        │    │ Audio Engine  │    │  Config        │  │
│  │  (PySide6)   │    │ (QSoundEffect│    │  Manager       │  │
│  │              │    │  + .wav pool) │    │  (JSON/TOML)   │  │
│  └──────▲───────┘    └──────▲───────┘    └───────▲────────┘  │
│         │                   │                    │            │
│         │ Signal/Slot       │ play(event)        │ load/save  │
│         │                   │                    │            │
│  ┌──────┴───────────────────┴────────────────────┴────────┐  │
│  │                  Game State Manager                     │  │
│  │                                                        │  │
│  │  - FSM (有限状態マシン)                                 │  │
│  │  - 認識結果の統合・フィルタリング                        │  │
│  │  - イベント発火 (Qt Signal)                             │  │
│  │  - 履歴ログ記録                                        │  │
│  └────────────────────▲───────────────────────────────────┘  │
│                       │                                      │
│                       │ RecognitionResult                    │
│                       │                                      │
│  ┌────────────────────┴───────────────────────────────────┐  │
│  │              Recognition Pipeline                       │  │
│  │                                                        │  │
│  │   ┌─────────────┐  ┌──────────────┐  ┌─────────────┐  │  │
│  │   │ Scene       │  │ Object       │  │ Digit       │  │  │
│  │   │ Classifier  │  │ Detector     │  │ Recognizer  │  │  │
│  │   │ (ONNX)      │  │ (ONNX)       │  │ (pHash/CNN) │  │  │
│  │   │             │  │              │  │             │  │  │
│  │   │ in: 224x224 │  │ in: 640x640  │  │ in: ROI crop│  │  │
│  │   │ out: label  │  │ out: boxes   │  │ out: digits │  │  │
│  │   └─────────────┘  └──────────────┘  └─────────────┘  │  │
│  │   ┌──────────────────────┐ ┌──────────────────────┐   │  │
│  │   │ Text Identifier      │ │ Feature Matcher      │   │  │
│  │   │ (pHash)              │ │ (ORB/AKAZE)          │   │  │
│  │   │ in: ROI crop         │ │ in: ROI crop         │   │  │
│  │   │ out: text_id + conf  │ │ out: match_id + conf │   │  │
│  │   └──────────────────────┘ └──────────────────────┘   │  │
│  └────────────────────▲───────────────────────────────────┘  │
│                       │                                      │
│                       │ numpy.ndarray (BGR frame)            │
│                       │                                      │
│  ┌────────────────────┴───────────────────────────────────┐  │
│  │                 Capture Module                          │  │
│  │                                                        │  │
│  │  - OBS Virtual Camera (cv2.VideoCapture)               │  │
│  │  - NDI Input (将来対応)                                 │  │
│  │  - フレームレート制御: 解析用=5-10fps, プレビュー用=30fps│  │
│  │  - 解像度正規化: 入力をFHD(1920x1080)に統一             │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### スレッドモデル

```
メインスレッド (Qt Event Loop)
  └── GUI描画、Signalの受信とSlot実行

キャプチャスレッド (QThread)
  └── カメラからフレーム取得 → 共有バッファに書き込み

認識ワーカースレッド (QThread)
  └── 共有バッファからフレーム読み取り → パイプライン実行 → Signal発火

音声スレッド
  └── QSoundEffect は Qt が内部管理（明示的スレッド不要）
```

フレームの受け渡しは `QMutex` + ダブルバッファリングで保護。
認識ワーカーが前フレーム処理中に新フレームが来た場合、最新フレームのみ保持（ドロップOK）。

---

## 4. 映像入力モジュール

### 4.1 OBS仮想カメラ入力

```python
# capture/obs_camera.py の基本設計

class OBSCameraCapture(QThread):
    frame_ready = Signal(np.ndarray)  # BGR frame (1920x1080)
    error_occurred = Signal(str)

    def __init__(self, device_index: int = 0, target_fps: float = 10.0):
        self.device_index = device_index
        self.target_fps = target_fps
        self._running = False

    def run(self):
        cap = cv2.VideoCapture(self.device_index)
        # Linux: V4L2 backend, Windows: DirectShow or MSMF
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        interval = 1.0 / self.target_fps
        while self._running:
            ret, frame = cap.read()
            if ret:
                # 解像度が異なる場合はリサイズ
                if frame.shape[:2] != (1080, 1920):
                    frame = cv2.resize(frame, (1920, 1080))
                self.frame_ready.emit(frame)
            time.sleep(interval)
```

### 4.2 カメラデバイス列挙

```python
def list_cameras() -> list[dict]:
    """利用可能なカメラデバイスを列挙"""
    cameras = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cameras.append({
                "index": i,
                "name": f"Camera {i}",  # OSによってはデバイス名取得可能
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            })
            cap.release()
    return cameras
```

### 4.3 解像度正規化の方針

入力がFHD以外の場合（720p, 4K等）、認識パイプラインに渡す前に1920x1080にリサイズ。
ROI座標をすべてFHD基準で定義することで、異なる入力解像度でも同一のパイプラインが動作する。

---

## 5. 認識パイプライン詳細

### 5.1 全体フロー

```
フレーム取得 (1920x1080 BGR)
    │
    ▼
┌─────────────────────────────────────┐
│ Stage 1: シーン分類                  │  ← 毎フレーム実行 (約2-5ms)
│ 入力: リサイズ(224x224)              │
│ 出力: SceneType enum + confidence   │
└──────────────┬──────────────────────┘
               │
     ┌─────────┼──────────┬──────────────────┐
     ▼         ▼          ▼                  ▼
   LOBBY    WAVE_ACTIVE  WAVE_RESULT      SPECIAL_WAVE
     │         │          │                  │
     │         ▼          ▼                  ▼
     │    ┌────────┐  ┌────────┐      ┌────────────┐
     │    │Stage 2 │  │Stage 3 │      │Stage 2     │
     │    │物体検出│  │数字認識│      │(特殊Wave用)│
     │    │        │  │+ S3b   │      │+ Stage 3   │
     │    └────────┘  │テキスト│      │+ Stage 3b  │
     │         │      │識別    │      └────────────┘
     │         ▼      └────────┘             │
     │    ┌────────┐      │                  │
     │    │Stage 4 │      │                  │
     │    │特徴量  │      │                  │
     │    │補助判定│      │                  │
     │    └────────┘      │                  │
     │         │          │                  │
     └─────────┴──────────┴──────────────────┘
               │
               ▼
         RecognitionResult (dataclass)
               │
               ▼
         Game State Manager
```

### 5.2 Stage 1: シーン分類 (Scene Classifier)

**目的**: 現在のゲーム画面がどのシーンかを判定する。後段の処理を分岐させるゲートの役割。

#### シーンラベル定義

```python
class SceneType(Enum):
    UNKNOWN = "unknown"                # 判定不能
    TITLE = "title"                    # タイトル画面
    LOBBY = "lobby"                    # ロビー（マッチング中含む）
    LOADING = "loading"                # ロード画面
    WAVE_INTRO = "wave_intro"          # Wave開始演出（潮位表示）
    WAVE_ACTIVE = "wave_active"        # Wave中（通常）
    WAVE_ACTIVE_HT = "wave_active_ht"  # Wave中（満潮 High Tide）
    WAVE_ACTIVE_LT = "wave_active_lt"  # Wave中（干潮 Low Tide）
    SPECIAL_RUSH = "special_rush"      # ヒカリバエ（ラッシュ）
    SPECIAL_GRILL = "special_grill"    # グリル発進
    SPECIAL_MOTHERSHIP = "special_mothership"  # ハコビヤ襲来
    SPECIAL_FOG = "special_fog"        # 霧
    SPECIAL_MUDMOUTH = "special_mudmouth"      # ドロシャケ噴出
    SPECIAL_COHOCK = "special_cohock"  # ドスコイ大量発生
    SPECIAL_GIANT = "special_giant"    # 巨大タツマキ(キング)
    WAVE_RESULT = "wave_result"        # Wave間リザルト
    GAME_CLEAR = "game_clear"          # クリアリザルト
    GAME_OVER = "game_over"            # 失敗リザルト
```

#### モデル仕様

- ベース: YOLOv8n-cls (Ultralytics)
- 入力: 224×224 RGB (正規化済み)
- 出力: クラス数 = 上記Enum数 (約18クラス)
- 推論: ONNX Runtime (CPU)、約2-5ms/フレーム

#### 環境差への対策（学習時のAugmentation）

```python
# tools/train_classifier.py での augmentation 設定例
augmentation = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=30, p=0.7),
    A.GaussNoise(var_limit=(5, 25), p=0.3),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.ImageCompression(quality_lower=70, quality_upper=100, p=0.3),  # 配信圧縮ノイズ
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

ポイント: **明るさ・コントラスト・色温度を積極的に変動させる**ことで、
モニター設定やキャプチャボードの差を学習時に吸収する。

#### 判定の安定化

```python
class SceneClassifier:
    def __init__(self, model_path: str, history_size: int = 5, threshold: float = 0.7):
        self.session = ort.InferenceSession(model_path)
        self.history: deque[SceneType] = deque(maxlen=history_size)
        self.threshold = threshold

    def classify(self, frame: np.ndarray) -> tuple[SceneType, float]:
        """単一フレーム分類"""
        input_tensor = self._preprocess(frame)
        outputs = self.session.run(None, {"images": input_tensor})
        probs = softmax(outputs[0][0])
        top_idx = np.argmax(probs)
        confidence = probs[top_idx]
        scene = SceneType(LABEL_MAP[top_idx])
        return scene, confidence

    def classify_stable(self, frame: np.ndarray) -> tuple[SceneType, float]:
        """多数決による安定化判定"""
        scene, conf = self.classify(frame)
        self.history.append(scene)
        if conf < self.threshold:
            return SceneType.UNKNOWN, conf
        # 直近N フレームの多数決
        counter = Counter(self.history)
        most_common, count = counter.most_common(1)[0]
        if count >= len(self.history) * 0.6:  # 60%以上一致
            return most_common, conf
        return SceneType.UNKNOWN, conf
```

### 5.3 Stage 2: オオモノシャケ検出 (Object Detector)

**目的**: Wave中にどのオオモノシャケが出現しているかを検出する。

#### 検出対象クラス

```python
class BossType(Enum):
    STEELHEAD = "bakudan"          # バクダン
    FLYFISH = "katagata"           # カタパッド
    STEEL_EEL = "hebi"             # ヘビ
    SCRAPPER = "teppan"            # テッパン
    STINGER = "tower"              # タワー
    MAWS = "mogura"                # モグラ
    DRIZZLER = "koumori"           # コウモリ
    FISH_STICK = "hashira"         # ハシラ
    FLIPPER_FLOPPER = "diver"      # ダイバー
    SLAMMIN_LID = "nabebuta"       # ナベブタ
    BIG_SHOT = "teppou"            # テッキュウ(大砲)
    # キングサーモン
    COHOZUNA = "yokozuna"          # ヨコヅナ
    HORRORBOROS = "tatsu"          # タツ
    MEGALODONTIA = "jaw"           # ジョー
```

#### モデル仕様

- ベース: YOLOv8n or YOLOv8s (Ultralytics)
- 入力: 640×640 RGB
- 出力: バウンディングボックス + クラスID + 信頼度
- 推論: ONNX Runtime (CPU)、約10-30ms/フレーム
- NMS閾値: 0.45、信頼度閾値: 0.5

#### Wave中のみ実行（条件付き実行）

```python
class RecognitionPipeline:
    def process(self, frame: np.ndarray) -> RecognitionResult:
        scene, scene_conf = self.scene_classifier.classify_stable(frame)
        result = RecognitionResult(scene=scene, scene_confidence=scene_conf)

        if scene in WAVE_ACTIVE_SCENES:
            # オオモノ検出はWave中のみ
            detections = self.object_detector.detect(frame)
            result.boss_detections = detections

            # 数字認識・テキスト識別も同時実行
            result.digits = self._read_digits(frame, scene)
            result.text_ids = self._identify_texts(frame, scene)

            # 特徴量マッチングで補助判定
            result.feature_matches = self.feature_matcher.match(frame, scene)

        elif scene in [SceneType.WAVE_RESULT, SceneType.GAME_CLEAR, SceneType.GAME_OVER]:
            # リザルト画面では数字認識のみ
            result.digits = self._read_digits(frame, scene)

        return result
```

### 5.4 Stage 3: 数字認識 (Digit Recognizer)

**目的**: 画面上の数値情報（金イクラ数、ノルマ、残り時間など）を読み取る。

> **設計判断: なぜ汎用OCR（PaddleOCR, EasyOCR, Tesseract）を使わないか**
>
> サーモンランのゲーム画面は汎用OCRが想定する環境と根本的に異なる:
> - フォント背景がダイナミックに動く（3Dゲーム画面）
> - ゲーム独自フォント（標準フォントと形状が異なる）
> - フォントが透過表示されている場合がある
> - 文字が水平に配置されていない（斜め・アーチ状など見栄え重視の配置）
>
> 実験の結果、汎用OCRでは実用的な精度が出ないことが確認されている。
> サーモンランの数字は**表示位置が固定・文字種が0-9と記号のみ**なので、
> 汎用OCRではなく**専用の分類器**で解くのが正しいアプローチ。

#### 認識手法: パーセプチュアルハッシュ (pHash) + フォールバックCNN

数字認識は「画像分類」として解く。出る場所・大きさ・文字種が既知なので、
ROI切り出し → 1文字ずつ分類 の流れで高精度・高速に認識できる。

**第一候補: pHash（パーセプチュアルハッシュ）**
- 画像の「構造的指紋」を64bitハッシュに変換して比較
- テンプレートマッチングの上位互換：明るさ・コントラスト変動に**非常に強い**
- 1回の比較がハミング距離計算（ビット演算）のみ → **1ms以下**
- メモリは1テンプレートあたり**たった8バイト**

**フォールバック: 軽量CNN**
- pHashで精度不足の場合に切り替え
- 32x32入力、12クラス（0-9, :, /）の超軽量モデル
- ONNX Runtime (CPU) で推論 2-3ms

#### pHashの仕組み

```
テンプレートマッチング（従来）:
  画像A のピクセル値 と 画像B のピクセル値 を直接比較
  → 明るさが変わると全ピクセルがズレる → 精度低下

パーセプチュアルハッシュ（pHash）:
  1. 画像を 32x32 にリサイズ
  2. グレースケール化
  3. DCT（離散コサイン変換）で周波数成分を抽出
  4. 低周波成分だけで 64bit ハッシュを生成
  5. ハミング距離で比較（ビット差の数）
  → 明るさ・コントラストが変わっても構造は変わらない → 精度維持
```

#### ROI（関心領域）定義

すべてFHD (1920x1080) 基準の座標。ゲーム画面のUI配置に基づく。

```python
# recognition/roi_definitions.py

@dataclass
class ROI:
    """FHD基準の関心領域"""
    x: int       # 左上X
    y: int       # 左上Y
    width: int   # 幅
    height: int  # 高さ
    name: str    # 識別名

# Wave中のROI定義
# ※ 座標は実際のゲーム画面のスクリーンショットから計測して調整すること
WAVE_ROIS = {
    "golden_egg_current": ROI(x=880, y=40, width=80, height=40, name="金イクラ現在数"),
    "golden_egg_quota":   ROI(x=970, y=40, width=60, height=40, name="金イクラノルマ"),
    "timer":              ROI(x=920, y=10, width=80, height=30, name="残り時間"),
    "wave_number":        ROI(x=60, y=40, width=120, height=40, name="Wave番号"),
}

# リザルト画面のROI定義
RESULT_ROIS = {
    "total_golden_eggs":  ROI(x=800, y=300, width=120, height=50, name="合計金イクラ"),
    "total_power_eggs":   ROI(x=800, y=360, width=120, height=50, name="合計赤イクラ"),
    "hazard_level":       ROI(x=800, y=420, width=100, height=40, name="キケン度"),
}
```

#### 数字分類器の実装

```python
class DigitRecognizer:
    """
    サーモンラン専用の数字認識器。
    ROI切り出し → 前処理 → 1文字ずつpHash比較 で数字列を認識する。
    """

    CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', '/']

    def __init__(self, template_dir: str, hash_threshold: int = 12):
        """
        Args:
            template_dir: 各文字のテンプレート画像ディレクトリ
                          0.png, 1.png, ..., 9.png, colon.png, slash.png
            hash_threshold: ハミング距離の閾値（小さいほど厳密）
        """
        self.hash_threshold = hash_threshold
        self.template_hashes: dict[str, int] = {}
        self._build_templates(template_dir)

    def _build_templates(self, template_dir: str):
        """テンプレート画像からpHashを事前計算"""
        for cls in self.CLASSES:
            filename = cls if cls.isdigit() else {":" : "colon", "/": "slash"}[cls]
            path = Path(template_dir) / f"{filename}.png"
            if path.exists():
                img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                self.template_hashes[cls] = self._compute_phash(img)

    def _compute_phash(self, img: np.ndarray, hash_size: int = 8) -> int:
        """パーセプチュアルハッシュを計算"""
        # 1. リサイズ（hash_size*4 の正方形 → DCTの入力）
        resized = cv2.resize(img, (hash_size * 4, hash_size * 4),
                             interpolation=cv2.INTER_AREA)
        # 2. float変換
        pixels = resized.astype(np.float64)
        # 3. DCT（離散コサイン変換）
        dct = cv2.dct(pixels)
        # 4. 左上の低周波成分だけ取得
        dct_low = dct[:hash_size, :hash_size]
        # 5. 中央値を閾値にしてビット化
        median = np.median(dct_low)
        diff = dct_low > median
        # 6. 64bitハッシュに変換
        return int(''.join(['1' if b else '0' for b in diff.flatten()]), 2)

    def _hamming_distance(self, hash1: int, hash2: int) -> int:
        """2つのハッシュのハミング距離（異なるビットの数）"""
        return bin(hash1 ^ hash2).count('1')

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """ゲーム画面のフォント用前処理"""
        if len(img.shape) == 3:
            # カラーマスク: ゲームフォントは特定色域
            # （白文字なら明度の高いピクセルを抽出）
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask = hsv[:, :, 2] > 180  # 閾値は実験で調整
            gray = (mask * 255).astype(np.uint8)
        else:
            gray = img
        return gray

    def _segment_characters(self, binary: np.ndarray) -> list[np.ndarray]:
        """1文字ずつ切り出す（等幅フォント前提）"""
        # 方法1: 等幅分割（ゲームフォントは等幅の場合が多い）
        h, w = binary.shape
        # 文字数を推定（幅/高さの比率から）
        char_width = h * 0.6  # 推定文字幅（実験で調整）
        n_chars = max(1, round(w / char_width))
        step = w / n_chars

        chars = []
        for i in range(n_chars):
            x1 = int(i * step)
            x2 = int((i + 1) * step)
            char_img = binary[:, x1:x2]
            # 空白（ほぼ黒）のセグメントはスキップ
            if np.mean(char_img) > 10:
                chars.append(char_img)
        return chars

    def recognize(self, roi_image: np.ndarray) -> tuple[str, float]:
        """
        ROI画像から数字列を読み取る。

        Returns:
            (認識結果の文字列, 平均信頼度)
        """
        processed = self._preprocess(roi_image)
        char_images = self._segment_characters(processed)

        result = ""
        total_confidence = 0

        for char_img in char_images:
            char_hash = self._compute_phash(char_img)

            best_char = "?"
            best_distance = 999

            for cls, tmpl_hash in self.template_hashes.items():
                dist = self._hamming_distance(char_hash, tmpl_hash)
                if dist < best_distance:
                    best_distance = dist
                    best_char = cls

            if best_distance <= self.hash_threshold:
                result += best_char
                # 信頼度: ハミング距離が0なら1.0、閾値なら0.0
                total_confidence += 1.0 - (best_distance / self.hash_threshold)
            else:
                result += "?"  # 認識失敗

        avg_confidence = total_confidence / len(char_images) if char_images else 0
        return result, avg_confidence
```

#### 数字認識結果の時系列フィルタリング

```python
class RecognitionSmoother:
    """認識結果の時系列スムージング（数字・テキスト共通）"""

    def __init__(self, history_size: int = 5):
        self.history: dict[str, deque] = defaultdict(lambda: deque(maxlen=history_size))

    def smooth(self, key: str, value: str | None) -> str | None:
        if value is None or "?" in (value or ""):
            return self._last_valid(key)

        self.history[key].append(value)

        # 数値の急激な変化を検知（外れ値フィルタ）
        values = list(self.history[key])
        if len(values) >= 3:
            nums = [int(v) for v in values if v and v.replace(":", "").replace("/", "").isdigit()]
            if len(nums) >= 3:
                median = sorted(nums)[len(nums) // 2]
                try:
                    current = int(value.replace(":", "").replace("/", ""))
                    if abs(current - median) > median * 0.5 and median > 0:
                        return str(median)
                except ValueError:
                    pass

        return value

    def _last_valid(self, key: str) -> str | None:
        for v in reversed(self.history[key]):
            if v is not None and "?" not in v:
                return v
        return None
```

### 5.4b Stage 3b: 固定テキスト識別 (Text Identifier)

**目的**: 画面上の固定テキスト（潮位名、特殊Wave名など）を識別する。

> これも汎用OCRではなく「どのテンプレートに最も近いか」の分類問題。
> 決まった文字列しか出ないので、全候補のpHashを事前計算して比較するだけ。

#### 識別対象

```python
# 固定テキストの候補一覧
FIXED_TEXTS = {
    # 潮位
    "tide_normal": "通常",
    "tide_high": "満潮",
    "tide_low": "干潮",
    # 特殊Wave
    "special_rush": "ヒカリバエ",
    "special_grill": "グリル発進",
    "special_mothership": "ハコビヤ襲来",
    "special_fog": "霧",
    "special_mudmouth": "ドロシャケ噴出",
    "special_cohock": "ドスコイ大量発生",
    # ステータス
    "clear": "CLEAR!!",
    "failure": "FAILURE",
    # Wave表示
    "wave1": "WAVE 1",
    "wave2": "WAVE 2",
    "wave3": "WAVE 3",
    "ex_wave": "EX-WAVE",
    # ... 必要に応じて追加
}
```

#### 実装

```python
class TextIdentifier:
    """
    固定テキスト識別器。
    テンプレート画像のpHashと比較して、最も近いテキストIDを返す。
    DigitRecognizerと同じpHash手法だが、1文字ずつではなく
    テキスト領域全体を1つのハッシュとして比較する。
    """

    def __init__(self, template_dir: str, hash_threshold: int = 15):
        self.hash_threshold = hash_threshold
        self.template_hashes: dict[str, int] = {}
        self._build_templates(template_dir)

    def _build_templates(self, template_dir: str):
        """テンプレート画像からpHashを事前計算"""
        for path in Path(template_dir).glob("*.png"):
            text_id = path.stem  # ファイル名がテキストID
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            self.template_hashes[text_id] = self._compute_phash(img)

    def identify(self, roi_image: np.ndarray) -> tuple[str | None, float]:
        """
        ROI画像からテキストIDを識別する。

        Returns:
            (テキストID or None, 信頼度)
        """
        if len(roi_image.shape) == 3:
            gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi_image

        query_hash = self._compute_phash(gray)

        best_id = None
        best_distance = 999

        for text_id, tmpl_hash in self.template_hashes.items():
            dist = self._hamming_distance(query_hash, tmpl_hash)
            if dist < best_distance:
                best_distance = dist
                best_id = text_id

        if best_distance <= self.hash_threshold:
            confidence = 1.0 - (best_distance / self.hash_threshold)
            return best_id, confidence
        return None, 0.0

    # _compute_phash, _hamming_distance は DigitRecognizer と共通
    # → 共通ユーティリティ (recognition/phash_utils.py) に切り出す
```

#### pHash共通ユーティリティ

DigitRecognizer と TextIdentifier で共有する:

```python
# recognition/phash_utils.py

def compute_phash(img: np.ndarray, hash_size: int = 8) -> int:
    """パーセプチュアルハッシュを計算"""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(img, (hash_size * 4, hash_size * 4),
                         interpolation=cv2.INTER_AREA)
    pixels = resized.astype(np.float64)
    dct = cv2.dct(pixels)
    dct_low = dct[:hash_size, :hash_size]
    median = np.median(dct_low)
    diff = dct_low > median
    return int(''.join(['1' if b else '0' for b in diff.flatten()]), 2)

def hamming_distance(hash1: int, hash2: int) -> int:
    """ハミング距離"""
    return bin(hash1 ^ hash2).count('1')
```

### 5.5 Stage 4: 特徴量マッチング (Feature Matcher)

**目的**: MLモデルの学習データが不足するUI要素の検出を補助する。

#### 主な検出対象

- 潮位アイコン（Wave開始演出中）
- スペシャルウェポンアイコン
- 特殊Wave演出のテキスト/アイコン
- 「たすけて！」シグナル

#### テンプレートマッチングとの違い

| | テンプレートマッチング | 特徴量マッチング (ORB/AKAZE) |
|---|---|---|
| 明るさ変動 | 弱い（直接影響） | 強い（特徴点ベース） |
| スケール変動 | 弱い | 中程度（AKAZEはマルチスケール） |
| 回転 | 弱い | 強い |
| 速度 | 高速 | やや遅い |
| 用途 | 固定位置・固定サイズ | 可変条件 |

#### 実装方針

```python
class FeatureMatcher:
    def __init__(self, template_dir: str, method: str = "akaze"):
        if method == "akaze":
            self.detector = cv2.AKAZE_create()
        else:
            self.detector = cv2.ORB_create(nfeatures=500)

        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.templates: dict[str, TemplateData] = {}
        self._load_templates(template_dir)

    def _load_templates(self, template_dir: str):
        """参照画像から特徴量を事前計算"""
        for path in Path(template_dir).glob("*.png"):
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            kp, desc = self.detector.detectAndCompute(img, None)
            self.templates[path.stem] = TemplateData(
                keypoints=kp,
                descriptors=desc,
                image=img,
            )

    def match(self, frame: np.ndarray, scene: SceneType) -> list[FeatureMatch]:
        """シーンに応じた参照画像とのマッチング"""
        results = []
        targets = self._get_targets_for_scene(scene)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for target_name in targets:
            template = self.templates.get(target_name)
            if template is None:
                continue

            # ROIで切り出し（検索範囲を限定して高速化）
            roi = self._get_roi_for_target(target_name)
            if roi:
                crop = gray[roi.y:roi.y+roi.height, roi.x:roi.x+roi.width]
            else:
                crop = gray

            kp, desc = self.detector.detectAndCompute(crop, None)
            if desc is None:
                continue

            matches = self.matcher.match(template.descriptors, desc)
            matches = sorted(matches, key=lambda m: m.distance)

            # 良好なマッチ数で判定
            good_matches = [m for m in matches if m.distance < 60]
            confidence = len(good_matches) / max(len(template.descriptors), 1)

            if confidence > 0.15:  # 閾値
                results.append(FeatureMatch(
                    name=target_name,
                    confidence=confidence,
                    match_count=len(good_matches),
                ))

        return results
```

### 5.6 パイプライン統合

```python
@dataclass
class RecognitionResult:
    """認識パイプラインの統合結果"""
    timestamp: float
    scene: SceneType
    scene_confidence: float
    boss_detections: list[BossDetection] = field(default_factory=list)
    digits: DigitData | None = None
    text_ids: dict[str, tuple[str, float]] = field(default_factory=dict)  # {roi_name: (text_id, confidence)}
    feature_matches: list[FeatureMatch] = field(default_factory=list)

@dataclass
class BossDetection:
    boss_type: BossType
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2

@dataclass
class DigitData:
    """数字認識の結果"""
    golden_egg_current: int | None = None
    golden_egg_quota: int | None = None
    timer_seconds: int | None = None
    timer_raw: str | None = None        # 生認識結果（デバッグ用）
    wave_number: int | None = None
    total_golden_eggs: int | None = None
    total_power_eggs: int | None = None
    hazard_level: str | None = None

@dataclass
class FeatureMatch:
    name: str
    confidence: float
    match_count: int

class RecognitionPipeline:
    """全Stageを統合するパイプライン"""

    def __init__(self, config: RecognitionConfig):
        self.scene_classifier = SceneClassifier(config.scene_model_path)
        self.object_detector = ObjectDetector(config.boss_model_path)
        self.digit_recognizer = DigitRecognizer(config.digit_template_dir)
        self.text_identifier = TextIdentifier(config.text_template_dir)
        self.smoother = RecognitionSmoother()
        self.feature_matcher = FeatureMatcher(config.feature_template_dir)

    def process(self, frame: np.ndarray) -> RecognitionResult:
        timestamp = time.time()

        # Stage 1: 常に実行
        scene, scene_conf = self.scene_classifier.classify_stable(frame)

        result = RecognitionResult(
            timestamp=timestamp,
            scene=scene,
            scene_confidence=scene_conf,
        )

        # Stage 2-4: シーンに応じて条件付き実行
        if scene in WAVE_ACTIVE_SCENES:
            result.boss_detections = self.object_detector.detect(frame)
            result.digits = self._read_digits(frame, scene)
            result.text_ids = self._identify_texts(frame, scene)
            result.feature_matches = self.feature_matcher.match(frame, scene)

        elif scene in RESULT_SCENES:
            result.digits = self._read_digits(frame, scene)

        elif scene == SceneType.WAVE_INTRO:
            result.text_ids = self._identify_texts(frame, scene)
            result.digits = self._read_digits(frame, scene)

        return result

    def _read_digits(self, frame: np.ndarray, scene: SceneType) -> DigitData:
        """ROIごとに数字認識 + スムージング"""
        rois = get_digit_rois_for_scene(scene)
        data = {}
        for key, roi in rois.items():
            crop = frame[roi.y:roi.y+roi.height, roi.x:roi.x+roi.width]
            raw, conf = self.digit_recognizer.recognize(crop)
            smoothed = self.smoother.smooth(key, raw)
            data[key] = smoothed
        return DigitData(**self._parse_digit_data(data))

    def _identify_texts(self, frame: np.ndarray, scene: SceneType) -> dict:
        """ROIごとに固定テキスト識別"""
        rois = get_text_rois_for_scene(scene)
        results = {}
        for key, roi in rois.items():
            crop = frame[roi.y:roi.y+roi.height, roi.x:roi.x+roi.width]
            text_id, conf = self.text_identifier.identify(crop)
            if text_id:
                results[key] = (text_id, conf)
        return results

# シーングループ定義
WAVE_ACTIVE_SCENES = {
    SceneType.WAVE_ACTIVE,
    SceneType.WAVE_ACTIVE_HT,
    SceneType.WAVE_ACTIVE_LT,
    SceneType.SPECIAL_RUSH,
    SceneType.SPECIAL_GRILL,
    SceneType.SPECIAL_MOTHERSHIP,
    SceneType.SPECIAL_FOG,
    SceneType.SPECIAL_MUDMOUTH,
    SceneType.SPECIAL_COHOCK,
    SceneType.SPECIAL_GIANT,
}

RESULT_SCENES = {
    SceneType.WAVE_RESULT,
    SceneType.GAME_CLEAR,
    SceneType.GAME_OVER,
}
```

### 5.7 パフォーマンス目標

| Stage | 処理時間目標 | 実行頻度 |
|---|---|---|
| Scene Classifier (ONNX) | < 5ms | 毎フレーム (10fps) |
| Object Detector (ONNX) | < 30ms | Wave中のみ (10fps) |
| Digit Recognizer (pHash) | < 3ms | Wave中+リザルト (10fps) |
| Text Identifier (pHash) | < 2ms | Wave開始演出時 |
| Feature Matcher (AKAZE) | < 20ms | 必要時のみ |
| **合計** | **< 60ms** | **10fps以上を維持** |

すべてCPU推論。GPU非搭載のPCでも動作可能にする。
pHash方式はPaddleOCR比で10倍以上高速・メモリ消費は1/100以下。

---

## 6. ゲーム状態管理 (FSM)

### 6.1 状態遷移図

```
                        ┌──────────────┐
                        │   STARTUP    │
                        │ (カメラ未接続)│
                        └──────┬───────┘
                               │ カメラ接続成功
                               ▼
                        ┌──────────────┐
                   ┌───►│   IDLE       │◄──────────────┐
                   │    │ (待機中)      │               │
                   │    └──────┬───────┘               │
                   │           │ LOBBY検知              │
                   │           ▼                        │
                   │    ┌──────────────┐               │
                   │    │   LOBBY      │               │
                   │    │ (マッチング中)│               │
                   │    └──────┬───────┘               │
                   │           │ LOADING or             │
                   │           │ WAVE_INTRO検知         │
                   │           ▼                        │
                   │    ┌──────────────┐               │
                   │    │ WAVE_INTRO   │               │
                   │    │ (Wave開始演出)│               │
                   │    │ → 潮位検知    │               │
                   │    │ → Wave番号読取│               │
                   │    └──────┬───────┘               │
                   │           │ WAVE_ACTIVE検知        │
                   │           ▼                        │
                   │    ┌──────────────┐               │
                   │    │ WAVE_ACTIVE  │◄──┐           │
                   │    │ (Wave中)      │   │           │
                   │    │              │   │           │
                   │    │ ・オオモノ追跡│   │           │
                   │    │ ・ノルマ監視  │   │           │
                   │    │ ・残時間監視  │   │           │
                   │    └──────┬───────┘   │           │
                   │           │            │           │
                   │      ┌────┴─────┐      │           │
                   │      ▼          ▼      │           │
                   │  WAVE_RESULT  SPECIAL   │           │
                   │      │        WAVE ─────┘           │
                   │      │                              │
                   │      ├─ 次Wave有り → WAVE_INTRO ─┐  │
                   │      │                           │  │
                   │      │                           ▼  │
                   │      │                    (ループ)   │
                   │      │                              │
                   │      ├─ GAME_CLEAR ─────────────────┘
                   │      │
                   │      └─ GAME_OVER ──────────────────┘
                   │
                   └── カメラ切断 or 長時間UNKNOWN
```

### 6.2 状態管理の実装

```python
class GameState(Enum):
    STARTUP = "startup"
    IDLE = "idle"
    LOBBY = "lobby"
    WAVE_INTRO = "wave_intro"
    WAVE_ACTIVE = "wave_active"
    WAVE_RESULT = "wave_result"
    GAME_CLEAR = "game_clear"
    GAME_OVER = "game_over"

@dataclass
class GameData:
    """現在のゲーム状態データ"""
    state: GameState = GameState.STARTUP
    wave_number: int = 0            # 1-3 (ExtraWave含むと4)
    wave_max: int = 3
    tide: str = "normal"            # "normal" | "high" | "low"
    special_wave: str | None = None # None or "rush", "grill", etc.

    golden_egg_current: int = 0
    golden_egg_quota: int = 0
    timer_seconds: int = 100

    active_bosses: list[BossDetection] = field(default_factory=list)

    # 累積データ
    total_golden_eggs: int = 0
    total_power_eggs: int = 0
    waves_cleared: int = 0

class GameStateManager(QObject):
    """ゲーム状態を管理し、状態変化時にSignalを発火"""

    # Qt Signals
    state_changed = Signal(GameState, GameState)          # old, new
    wave_started = Signal(int, str, str)                  # wave_num, tide, special
    wave_cleared = Signal(int)                            # wave_num
    boss_appeared = Signal(str, float)                    # boss_type, confidence
    boss_disappeared = Signal(str)                        # boss_type
    quota_reached = Signal(int)                           # wave_num
    timer_warning = Signal(int)                           # remaining_seconds
    game_cleared = Signal()
    game_over = Signal()

    def __init__(self):
        super().__init__()
        self.data = GameData()
        self._previous_bosses: set[str] = set()
        self._quota_reached_flag = False
        self._timer_warnings_fired: set[int] = set()

    def update(self, result: RecognitionResult):
        """認識結果を受け取り、状態を更新"""
        new_state = self._determine_state(result)

        if new_state != self.data.state:
            old_state = self.data.state
            self.data.state = new_state
            self.state_changed.emit(old_state, new_state)
            self._on_state_transition(old_state, new_state, result)

        if new_state == GameState.WAVE_ACTIVE:
            self._update_wave_data(result)

    def _determine_state(self, result: RecognitionResult) -> GameState:
        """認識結果からゲーム状態を決定"""
        scene = result.scene
        scene_to_state = {
            SceneType.LOBBY: GameState.LOBBY,
            SceneType.WAVE_INTRO: GameState.WAVE_INTRO,
            SceneType.WAVE_RESULT: GameState.WAVE_RESULT,
            SceneType.GAME_CLEAR: GameState.GAME_CLEAR,
            SceneType.GAME_OVER: GameState.GAME_OVER,
        }
        if scene in WAVE_ACTIVE_SCENES:
            return GameState.WAVE_ACTIVE
        return scene_to_state.get(scene, self.data.state)  # 不明時は現状維持

    def _on_state_transition(self, old: GameState, new: GameState, result: RecognitionResult):
        """状態遷移時の処理"""
        if new == GameState.WAVE_ACTIVE and old in (GameState.WAVE_INTRO, GameState.LOBBY):
            self.data.wave_number += 1
            self._quota_reached_flag = False
            self._timer_warnings_fired.clear()
            tide = self._detect_tide(result)
            special = self._detect_special(result)
            self.data.tide = tide
            self.data.special_wave = special
            self.wave_started.emit(self.data.wave_number, tide, special or "normal")

        elif new == GameState.WAVE_RESULT and old == GameState.WAVE_ACTIVE:
            self.data.waves_cleared += 1
            self.wave_cleared.emit(self.data.wave_number)

        elif new == GameState.GAME_CLEAR:
            self.game_cleared.emit()
            self._reset_game()

        elif new == GameState.GAME_OVER:
            self.game_over.emit()
            self._reset_game()

    def _update_wave_data(self, result: RecognitionResult):
        """Wave中のデータ更新"""
        if result.ocr_data:
            # ノルマ
            if result.ocr_data.golden_egg_current:
                try:
                    self.data.golden_egg_current = int(result.ocr_data.golden_egg_current)
                except ValueError:
                    pass
            if result.ocr_data.golden_egg_quota:
                try:
                    self.data.golden_egg_quota = int(result.ocr_data.golden_egg_quota)
                except ValueError:
                    pass

            # ノルマ達成チェック
            if (not self._quota_reached_flag
                and self.data.golden_egg_quota > 0
                and self.data.golden_egg_current >= self.data.golden_egg_quota):
                self._quota_reached_flag = True
                self.quota_reached.emit(self.data.wave_number)

            # 残り時間
            if result.ocr_data.timer:
                try:
                    self.data.timer_seconds = self._parse_timer(result.ocr_data.timer)
                    # 残り時間警告
                    for threshold in [30, 15, 10]:
                        if (self.data.timer_seconds <= threshold
                            and threshold not in self._timer_warnings_fired):
                            self._timer_warnings_fired.add(threshold)
                            self.timer_warning.emit(threshold)
                except ValueError:
                    pass

        # オオモノ出現/退場の差分検知
        current_bosses = {d.boss_type.value for d in result.boss_detections}
        appeared = current_bosses - self._previous_bosses
        disappeared = self._previous_bosses - current_bosses

        for boss in appeared:
            det = next(d for d in result.boss_detections if d.boss_type.value == boss)
            self.boss_appeared.emit(boss, det.confidence)
        for boss in disappeared:
            self.boss_disappeared.emit(boss)

        self._previous_bosses = current_bosses
        self.data.active_bosses = result.boss_detections

    def _reset_game(self):
        """ゲーム終了時のリセット"""
        self.data = GameData(state=GameState.IDLE)
        self._previous_bosses.clear()
```

---

## 7. 音声通知システム

### 7.1 音声ファイル構成

```
assets/sounds/
├── wave_start_1.wav        # 「Wave 1 開始」
├── wave_start_2.wav
├── wave_start_3.wav
├── wave_start_extra.wav    # 「EX Wave 開始」
├── tide_high.wav           # 「満潮です」
├── tide_low.wav            # 「干潮です」
├── tide_normal.wav         # 「通常潮位です」
├── special_rush.wav        # 「ラッシュです」
├── special_grill.wav       # 「グリル発進です」
├── special_mothership.wav  # 「ハコビヤ襲来です」
├── special_fog.wav         # 「霧が出ています」
├── special_mudmouth.wav    # 「ドロシャケ噴出です」
├── special_cohock.wav      # 「ドスコイ大量発生です」
├── boss/
│   ├── bakudan.wav         # 「バクダン」
│   ├── katagata.wav        # 「カタパッド」
│   ├── hebi.wav            # 「ヘビ」
│   ├── teppan.wav          # 「テッパン」
│   ├── tower.wav           # 「タワー」
│   ├── mogura.wav          # 「モグラ」
│   ├── koumori.wav         # 「コウモリ」
│   ├── hashira.wav         # 「ハシラ」
│   ├── diver.wav           # 「ダイバー」
│   ├── nabebuta.wav        # 「ナベブタ」
│   └── teppou.wav          # 「テッキュウ」
├── quota_reached.wav       # 「ノルマ達成」
├── timer_30.wav            # 「残り30秒」
├── timer_15.wav            # 「残り15秒」
├── timer_10.wav            # 「残り10秒」
├── game_clear.wav          # 「クリアです、お疲れさまでした」
└── game_over.wav           # 「失敗です」
```

### 7.2 音声エンジン実装

```python
class AudioNotifier(QObject):
    """音声通知エンジン"""

    def __init__(self, sounds_dir: str, parent=None):
        super().__init__(parent)
        self.sounds_dir = Path(sounds_dir)
        self.volume = 0.8  # 0.0 - 1.0
        self._sound_cache: dict[str, QSoundEffect] = {}
        self._queue: deque[str] = deque()
        self._is_playing = False
        self._preload_all()

    def _preload_all(self):
        """起動時に全音声ファイルをメモリにロード"""
        for wav_path in self.sounds_dir.rglob("*.wav"):
            key = str(wav_path.relative_to(self.sounds_dir).with_suffix(""))
            sound = QSoundEffect(self)
            sound.setSource(QUrl.fromLocalFile(str(wav_path)))
            sound.setVolume(self.volume)
            sound.playingChanged.connect(self._on_playing_changed)
            self._sound_cache[key] = sound

    def play(self, sound_key: str):
        """音声を再生キューに追加"""
        if sound_key in self._sound_cache:
            self._queue.append(sound_key)
            if not self._is_playing:
                self._play_next()

    def _play_next(self):
        if self._queue:
            key = self._queue.popleft()
            sound = self._sound_cache[key]
            sound.setVolume(self.volume)
            self._is_playing = True
            sound.play()
        else:
            self._is_playing = False

    def _on_playing_changed(self):
        sender = self.sender()
        if sender and not sender.isPlaying():
            self._play_next()

    def set_volume(self, volume: float):
        self.volume = max(0.0, min(1.0, volume))

    def stop_all(self):
        self._queue.clear()
        for sound in self._sound_cache.values():
            sound.stop()
        self._is_playing = False
```

### 7.3 GameStateManagerとの接続

```python
# app.py での接続例

def connect_audio(state_manager: GameStateManager, audio: AudioNotifier):
    state_manager.wave_started.connect(
        lambda num, tide, special: _on_wave_start(audio, num, tide, special)
    )
    state_manager.boss_appeared.connect(
        lambda boss, conf: audio.play(f"boss/{boss}")
    )
    state_manager.quota_reached.connect(
        lambda _: audio.play("quota_reached")
    )
    state_manager.timer_warning.connect(
        lambda secs: audio.play(f"timer_{secs}")
    )
    state_manager.game_cleared.connect(
        lambda: audio.play("game_clear")
    )
    state_manager.game_over.connect(
        lambda: audio.play("game_over")
    )

def _on_wave_start(audio: AudioNotifier, num: int, tide: str, special: str):
    wave_key = f"wave_start_{num}" if num <= 3 else "wave_start_extra"
    audio.play(wave_key)
    # 潮位
    audio.play(f"tide_{tide}")
    # 特殊Wave
    if special != "normal":
        audio.play(f"special_{special}")
```

### 7.4 VOICEVOX音声生成スクリプト

```python
# tools/generate_voices.py
# 事前にVOICEVOXエンジンをローカルで起動しておく必要がある
# https://voicevox.hiroshiba.jp/

import requests
from pathlib import Path

VOICEVOX_URL = "http://localhost:50021"
SPEAKER_ID = 3  # ずんだもん（ノーマル）

VOICE_TEXTS = {
    "wave_start_1": "ウェーブ1、開始",
    "wave_start_2": "ウェーブ2、開始",
    "wave_start_3": "ウェーブ3、開始",
    "wave_start_extra": "エクストラウェーブ、開始",
    "tide_high": "満潮です",
    "tide_low": "干潮です",
    "tide_normal": "通常潮位です",
    "special_rush": "ラッシュです、気をつけて",
    "special_grill": "グリル発進です",
    "special_mothership": "ハコビヤ襲来です",
    "special_fog": "霧が出ています",
    "special_mudmouth": "ドロシャケ噴出です",
    "special_cohock": "ドスコイ大量発生です",
    "boss/bakudan": "バクダン",
    "boss/katagata": "カタパッド",
    "boss/hebi": "ヘビ",
    "boss/teppan": "テッパン",
    "boss/tower": "タワー",
    "boss/mogura": "モグラ",
    "boss/koumori": "コウモリ",
    "boss/hashira": "ハシラ",
    "boss/diver": "ダイバー",
    "boss/nabebuta": "ナベブタ",
    "boss/teppou": "テッキュウ",
    "quota_reached": "ノルマ達成",
    "timer_30": "残り30秒",
    "timer_15": "残り15秒",
    "timer_10": "残り10秒",
    "game_clear": "クリアです、お疲れさまでした",
    "game_over": "失敗です",
}

def generate_voice(text: str, output_path: str, speaker_id: int = SPEAKER_ID):
    # 音声合成用クエリ作成
    query = requests.post(
        f"{VOICEVOX_URL}/audio_query",
        params={"text": text, "speaker": speaker_id},
    ).json()

    # 音声合成
    audio = requests.post(
        f"{VOICEVOX_URL}/synthesis",
        params={"speaker": speaker_id},
        json=query,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(audio.content)

def main():
    output_dir = Path("assets/sounds")
    for key, text in VOICE_TEXTS.items():
        output_path = output_dir / f"{key}.wav"
        if not output_path.exists():
            print(f"Generating: {key} -> {text}")
            generate_voice(text, str(output_path))
    print("Done!")

if __name__ == "__main__":
    main()
```

---

## 8. GUIデザイン仕様

### 8.1 メインウィンドウレイアウト

```
┌────────────────────────────────────────────────────┐
│  🐟 shakeop                    [_][□][X]       │
├────────────────────────────────────────────────────┤
│  カメラ: [ OBS Virtual Camera ▼ ]  [ ▶ 開始 ]     │
├────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────┐  │
│  │           ゲームステータスパネル              │  │
│  │                                              │  │
│  │  状態: 🟢 Wave 2 / 通常潮                    │  │
│  │                                              │  │
│  │  🥚 金イクラ: 14 / 18                        │  │
│  │  ⏱️  残り時間: 0:45                          │  │
│  │                                              │  │
│  │  出現中オオモノ:                              │  │
│  │  ┌──────┐ ┌──────┐ ┌──────┐                  │  │
│  │  │バクダン│ │テッパン│ │ヘビ  │                  │  │
│  │  │ 96%  │ │ 92%  │ │ 88%  │                  │  │
│  │  └──────┘ └──────┘ └──────┘                  │  │
│  └──────────────────────────────────────────────┘  │
│                                                    │
│  ┌──────────────────────────────────────────────┐  │
│  │  イベントログ                    [クリア]     │  │
│  │  12:03:15  Wave 2 開始 - 通常潮              │  │
│  │  12:03:18  バクダン 出現                     │  │
│  │  12:03:22  テッパン 出現                     │  │
│  │  12:03:35  ヘビ 出現                         │  │
│  │  12:03:42  ノルマ達成 (18/18)                │  │
│  └──────────────────────────────────────────────┘  │
│                                                    │
├────────────────────────────────────────────────────┤
│  🔊 [████████░░] 80%    認識: Scene 98% Digit 95%  │
│  [⚙ 設定]  [📌 常に手前]  [最小化モード]          │
└────────────────────────────────────────────────────┘
```

### 8.2 コンパクトモード（最小化表示）

```
┌─────────────────────────────┐
│ 🐟 W2 通常 🥚14/18 ⏱0:45  │
│ バクダン テッパン ヘビ       │
└─────────────────────────────┘
```

ゲーム画面の横やセカンドモニターに置けるサイズ。Always on Top対応。

### 8.3 設定画面

```
┌──────────────────────────────────────┐
│  ⚙ 設定                             │
├──────────────────────────────────────┤
│                                      │
│  [一般]  [認識]  [音声]  [表示]      │
│                                      │
│  ── 一般 ──                          │
│  解析FPS: [ 10 ▼ ]                   │
│  ログ保存: [✓] 有効                  │
│                                      │
│  ── 認識 ──                          │
│  シーン分類閾値: [0.7  ]             │
│  オオモノ検出閾値: [0.5  ]           │
│  認識スムージング: [5 フレーム]       │
│                                      │
│  ── 音声 ──                          │
│  音量: [████████░░] 80%              │
│  音声セット: [ ずんだもん ▼ ]        │
│  通知: [✓] Wave開始                  │
│        [✓] オオモノ出現              │
│        [✓] ノルマ達成                │
│        [✓] 残り時間                  │
│                                      │
│  ── 表示 ──                          │
│  常に手前に表示: [✓]                 │
│  起動時にコンパクトモード: [ ]       │
│  テーマ: [ ダーク ▼ ]                │
│                                      │
│          [保存]  [キャンセル]         │
└──────────────────────────────────────┘
```

---

## 9. 設定・データモデル

### 9.1 設定ファイル (config.toml)

```toml
[general]
analysis_fps = 10
log_enabled = true
log_dir = "logs"

[capture]
device_index = 0
resolution = [1920, 1080]

[recognition]
scene_model_path = "models/scene_classifier.onnx"
boss_model_path = "models/boss_detector.onnx"
digit_template_dir = "assets/templates/digits"
text_template_dir = "assets/templates/texts"
feature_template_dir = "assets/templates/features"
scene_threshold = 0.7
boss_threshold = 0.5
digit_hash_threshold = 12
text_hash_threshold = 15
smoothing_frames = 5

[audio]
enabled = true
volume = 0.8
sound_dir = "assets/sounds"
notify_wave_start = true
notify_boss_appear = true
notify_quota_reached = true
notify_timer_warning = true

[display]
always_on_top = false
start_compact = false
theme = "dark"
```

### 9.2 設定管理

```python
@dataclass
class AppConfig:
    """アプリ設定（TOML読み書き）"""
    # ... フィールド定義（config.tomlと対応）

    @classmethod
    def load(cls, path: str = "config.toml") -> "AppConfig":
        """設定ファイル読み込み。なければデフォルト生成。"""
        ...

    def save(self, path: str = "config.toml"):
        """設定ファイル保存"""
        ...
```

---

## 10. ディレクトリ構成

```
salmon-buddy/
├── .github/
│   └── workflows/
│       ├── ci.yml              # lint + test
│       └── build.yml           # PyInstaller ビルド
├── LICENSE                     # MIT
├── README.md
├── CLAUDE.md                   # Claude Code用プロジェクトガイド
├── pyproject.toml              # uv 用
├── uv.lock
│
├── src/
│   └── salmon_buddy/
│       ├── __init__.py
│       ├── __main__.py         # python -m salmon_buddy
│       ├── main.py             # エントリーポイント
│       ├── app.py              # QApplication + 全体配線
│       │
│       ├── capture/
│       │   ├── __init__.py
│       │   ├── base.py         # AbstractCapture
│       │   └── obs_camera.py
│       │
│       ├── recognition/
│       │   ├── __init__.py
│       │   ├── pipeline.py     # RecognitionPipeline
│       │   ├── scene_classifier.py
│       │   ├── object_detector.py
│       │   ├── digit_recognizer.py
│       │   ├── text_identifier.py
│       │   ├── recognition_smoother.py
│       │   ├── feature_matcher.py
│       │   ├── phash_utils.py  # pHash共通ユーティリティ
│       │   └── roi_definitions.py
│       │
│       ├── game_state/
│       │   ├── __init__.py
│       │   ├── state_machine.py  # GameStateManager
│       │   ├── events.py         # イベント定義
│       │   └── models.py         # GameData, BossDetection etc.
│       │
│       ├── audio/
│       │   ├── __init__.py
│       │   └── notifier.py       # AudioNotifier
│       │
│       ├── gui/
│       │   ├── __init__.py
│       │   ├── main_window.py
│       │   ├── status_panel.py
│       │   ├── log_panel.py
│       │   ├── settings_dialog.py
│       │   ├── compact_widget.py
│       │   └── styles/
│       │       ├── dark.qss
│       │       └── light.qss
│       │
│       └── config/
│           ├── __init__.py
│           └── settings.py       # AppConfig
│
├── models/                       # ONNXモデル（gitignore、別途配布）
│   ├── .gitkeep
│   ├── scene_classifier.onnx
│   └── boss_detector.onnx
│
├── assets/
│   ├── sounds/                   # VOICEVOX生成済み.wav
│   │   ├── boss/
│   │   └── ...
│   ├── icons/
│   │   └── app_icon.png
│   └── templates/                # pHash / 特徴量マッチング用参照画像
│       ├── digits/               # 数字テンプレート (0.png - 9.png, colon.png, slash.png)
│       ├── texts/                # 固定テキストテンプレート (tide_high.png, special_rush.png, ...)
│       └── features/             # AKAZE特徴量マッチング用参照画像
│
├── tools/
│   ├── generate_voices.py        # VOICEVOX音声生成
│   ├── collect_training_data.py  # 学習データ収集
│   ├── train_classifier.py       # シーン分類モデル学習
│   ├── train_detector.py         # オオモノ検出モデル学習
│   ├── export_onnx.py            # ONNX変換
│   └── roi_calibrator.py         # ROI座標調整ツール
│
├── tests/
│   ├── test_recognition.py
│   ├── test_game_state.py
│   ├── test_digit_recognizer.py
│   ├── test_text_identifier.py
│   └── fixtures/                 # テスト用スクリーンショット
│
├── docs/
│   ├── issues/                   # 管理番号別ドキュメント
│   │   ├── index.md              # 全管理番号一覧
│   │   ├── F-001_scene_classify/
│   │   │   ├── requirements.md   # 要求仕様書
│   │   │   └── design.md         # 機能設計書
│   │   ├── F-002_digit_recognition/
│   │   │   ├── requirements.md
│   │   │   └── design.md
│   │   └── ...
│   ├── ARCHITECTURE.md           # このドキュメントの要約版
│   └── TRAINING_GUIDE.md         # モデル学習手順
```

---

## 11. 開発フェーズとタスク

### Phase 1: 基盤 (MVP) — 目安2〜3週間

**ゴール**: カメラ映像を取得してGUIに表示、ボタンを押すと音が鳴る。

- [ ] `uv init` でプロジェクト初期化
- [ ] pyproject.toml に依存関係定義 (PySide6, opencv-python, onnxruntime, numpy)
- [ ] CLAUDE.md 作成（Claude Code用）
- [ ] capture/obs_camera.py — OBS仮想カメラ取得
- [ ] gui/main_window.py — 最小GUI（カメラ選択、開始/停止、プレビュー表示）
- [ ] audio/notifier.py — QSoundEffectによる音声再生
- [ ] tools/generate_voices.py — VOICEVOX音声生成スクリプト
- [ ] config/settings.py — TOML設定読み書き
- [ ] main.py + __main__.py — エントリーポイント

### Phase 2: シーン分類 + 数字/テキスト認識 — 目安3〜4週間

**ゴール**: シーンを自動判定し、数値・テキストをpHashで読み取り、GUIに表示する。

- [ ] tools/collect_training_data.py — フレーム自動収集ツール
- [ ] tools/roi_calibrator.py — ROI位置の視覚的調整ツール
- [ ] シーン分類の学習データ収集・ラベリング（最低各クラス100枚〜）
- [ ] tools/train_classifier.py + export_onnx.py
- [ ] recognition/scene_classifier.py
- [ ] recognition/phash_utils.py — pHash共通ユーティリティ
- [ ] recognition/digit_recognizer.py — 数字テンプレート作成 + pHash分類
- [ ] recognition/text_identifier.py — 固定テキストテンプレート作成 + pHash分類
- [ ] recognition/recognition_smoother.py — 時系列フィルタリング
- [ ] recognition/roi_definitions.py
- [ ] recognition/pipeline.py（Stage 1 + 3 + 3b 統合）
- [ ] game_state/state_machine.py — FSM基本実装
- [ ] game_state/models.py — データクラス
- [ ] gui/status_panel.py — ステータス表示
- [ ] gui/log_panel.py — イベントログ
- [ ] 音声通知の接続（Wave開始、ノルマ達成、残り時間）

### Phase 3: オオモノ検出 + 特徴量マッチング — 目安2〜3週間

**ゴール**: オオモノシャケを検出して音声通知する。

- [ ] オオモノの学習データ収集 + アノテーション（LabelImg / Roboflow等）
- [ ] tools/train_detector.py（YOLOv8 物体検出）
- [ ] recognition/object_detector.py
- [ ] recognition/feature_matcher.py（潮位アイコン、特殊Wave検知の補助）
- [ ] パイプライン完全統合（Stage 1-4）
- [ ] boss_appeared / boss_disappeared の音声通知接続
- [ ] 検出精度のチューニング

### Phase 4: 仕上げ — 目安2週間

**ゴール**: 配布可能な品質に仕上げる。

- [ ] gui/compact_widget.py — コンパクトモード
- [ ] gui/settings_dialog.py — 設定画面
- [ ] styles/ — ダーク/ライトテーマ
- [ ] Always on Top 機能
- [ ] PyInstaller でのパッケージング（Windows .exe, Linux AppImage）
- [ ] CI/CD（GitHub Actions: lint, test, build）
- [ ] README.md 整備
- [ ] テスト追加

### Phase 5: 拡張（オプション）

- [ ] NDI入力対応
- [ ] OBS WebSocket連携（配信オーバーレイ出力）
- [ ] 戦績記録・統計ダッシュボード
- [ ] イカリング3 API連携（武器・ステージ情報取得）
- [ ] ユーザーがカスタム音声セットを追加できる仕組み
- [ ] コミュニティによるモデル改善の受付フロー
- [ ] 武器アイコン分類（後述の付録B参照）
- [ ] pHashで精度不足の認識対象がある場合、軽量CNN分類器にフォールバック

---

## 12. モデル学習ガイド

### 12.1 シーン分類モデル

```bash
# 1. データ収集（ゲームプレイ中に自動保存）
python -m tools.collect_training_data --camera 0 --interval 1.0 --output data/raw

# 2. ラベリング（フォルダ分けによる分類）
# data/classified/
#   lobby/
#   wave_active/
#   wave_active_ht/
#   ...
# 手動でフォルダに仕分けする（またはラベリングツール使用）

# 3. 学習
python -m tools.train_classifier \
    --data data/classified \
    --model yolov8n-cls \
    --epochs 100 \
    --imgsz 224

# 4. ONNX変換
python -m tools.export_onnx --weights runs/classify/train/weights/best.pt --format onnx
```

### 12.2 オオモノ検出モデル

```bash
# 1. データ収集（Wave中のフレーム）
# collect_training_dataで収集したWave中フレームを使用

# 2. アノテーション
# LabelImg, Roboflow, CVAT 等でバウンディングボックスを付与
# YOLO形式で出力: class_id cx cy w h (正規化座標)

# 3. 学習
python -m tools.train_detector \
    --data data/boss_detection/data.yaml \
    --model yolov8n \
    --epochs 200 \
    --imgsz 640

# 4. ONNX変換
python -m tools.export_onnx --weights runs/detect/train/weights/best.pt --format onnx
```

### 12.3 データ拡張の重要性

明るさ・コントラスト・色温度のバリエーションを学習データに含めることが、
テンプレートマッチングに対する最大のアドバンテージ。

推奨augmentation:
- RandomBrightnessContrast (±30%)
- HueSaturationValue (hue±10, sat±30, val±30)
- GaussNoise (配信ノイズの模倣)
- ImageCompression (JPEG圧縮アーティファクトの模倣)
- MotionBlur (カメラ遅延の模倣)

---

## 13. クロスプラットフォーム対応メモ

### Windows

- OBS仮想カメラ: DirectShow backend (`cv2.VideoCapture(index, cv2.CAP_DSHOW)`)
- ONNX Runtime: `onnxruntime` パッケージ（GPU版は `onnxruntime-directml`）
- 配布: PyInstaller で .exe + フォルダ一式 or Inno Setup でインストーラー

### Linux

- OBS仮想カメラ: V4L2 backend（`v4l2loopback` モジュール必要）
- ONNX Runtime: `onnxruntime`
- PySide6: `libxcb` 等のシステムライブラリ依存あり
- 配布: AppImage or Flatpak

### 共通注意点

- `Path` を使い、パス区切り文字をハードコードしない
- カメラデバイス列挙方法がOS間で異なる（抽象化する）
- QSoundEffect は `.wav` のみ対応（mp3不可）。44.1kHz/16bit PCM推奨
- 設定ファイルの保存場所: `QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)`

---

## 付録A: CLAUDE.md テンプレート

```markdown
# CLAUDE.md — shakeop

## プロジェクト概要
Splatoon 3 サーモンラン NEXT WAVE のリアルタイム解析オペレーターアプリ。

## 技術スタック
- Python 3.11+ / uv
- PySide6 (GUI)
- OpenCV + ONNX Runtime (認識)
- pHash (数字・固定テキスト認識) — 汎用OCR不使用
- QSoundEffect (音声)

## 環境セットアップ
uv sync

## 実行
uv run python -m salmon_buddy

## テスト
uv run pytest

## コーディング規約
- フォーマッター: ruff format
- リンター: ruff check
- 型チェック: mypy --strict src/

## 重要な設計判断
- 設計ドキュメント: docs/ARCHITECTURE.md を参照
- 認識パイプラインは5段構成（Scene→Object→Digit→Text→Feature）
- 数字・テキスト認識にはpHash（パーセプチュアルハッシュ）を使用
  - 汎用OCRはゲーム画面（独自フォント・動く背景・透過・斜め配置）に不向き
- ゲーム状態はFSMで管理、Qt SignalでGUI/音声に伝播
- すべてのROI座標はFHD(1920x1080)基準
- ONNXモデルはCPU推論前提（GPU不要）

## ファイル構成の方針
- src/salmon_buddy/ 以下にモジュール分割
- 各モジュールは疎結合（Signal/Slotで接続）
- models/ はgitignore（ONNXファイルは別途配布）
```

---

## 付録B: 武器アイコン分類（将来対応メモ）

> 優先度: 低。基本機能が安定してから着手する。

### 概要

武器アイコンをゲーム画面から認識して武器名を判定する機能。
約120種類（亜種含む368ファイルの公式2Dアイコンが Inkipedia で入手可能）。

### 課題

- 亜種のアイコンが非常に似ている（例: スプラシューター vs ヒーローシューター レプリカ）
- ゲーム画面上ではアイコンが小さく、背景も変化する

### 候補手法

| 手法 | 亜種区別 | 学習データ | 備考 |
|---|---|---|---|
| pHash | △ 苦手 | 不要 | まず試す価値はある |
| 軽量CNN分類 (MobileNet等) | ○ | 各50-100枚 | 確実だが学習必要 |
| メトリック学習 (Triplet Loss) | ◎ | 各5-10枚 | 少量データで亜種区別に強い |

### ミニアプリ計画

```
exp_008_weapon_classify  — 武器アイコン分類実験
```

公式アイコン画像を参照ベクトルとして保持し、ゲーム画面から切り出したアイコンを
最も近い参照に分類するアプローチ（メトリック学習ベース）が有力。
