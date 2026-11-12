import csv
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import label_comments as labels


def response(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class LabelCommentsTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        self.input = root / '01.csv'
        self.output = root / '05.csv'
        self.texts = ['Hỏi lịch học', 'Mình pass sách']
        with self.input.open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['content'])
            writer.writerows((text,) for text in self.texts)
        self.addCleanup(patch.stopall)
        patch.multiple(labels, INPUT=self.input, OUTPUT=self.output, API_KEY='test-key').start()
        patch('sys.stdout', new_callable=io.StringIO).start()
        self.api = patch.object(labels, 'OpenAI').start()
        self.create = self.api.return_value.chat.completions.create

    def test_dry_run_needs_no_key_and_writes_nothing(self):
        with patch.object(labels, 'API_KEY', ''):
            labels.main(dry_run=True)
        self.api.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_labels_all_raw_rows_in_order_and_resumes(self):
        self.create.return_value = response('{"spam": [0]}')
        labels.main(limit=1)
        self.assertEqual(labels.read_labels(self.output), {'Hỏi lịch học': 'spam'})
        self.create.return_value = response('{"spam": []}')
        labels.main()
        self.assertEqual(labels.read_labels(self.output), {
            'Hỏi lịch học': 'spam', 'Mình pass sách': 'ham',
        })
        self.assertEqual(list(labels.read_labels(self.output)), self.texts)
        sent = [call.kwargs['messages'][1]['content'] for call in self.create.call_args_list]
        self.assertEqual(sent, [
            '{"comments":[[0,"Hỏi lịch học"]]}',
            '{"comments":[[0,"Mình pass sách"]]}',
        ])
        self.api.reset_mock()
        labels.main()
        self.api.assert_not_called()

    def test_invalid_response_splits_then_marks_review(self):
        self.create.side_effect = [response('invalid'), response('invalid'), response('{"spam": [0]}')]
        with self.assertRaisesRegex(SystemExit, 'review'):
            labels.main()
        self.assertEqual(labels.read_labels(self.output), {
            'Hỏi lịch học': 'review', 'Mình pass sách': 'spam',
        })
        with self.assertRaisesRegex(SystemExit, 'review'):
            labels.main()

    def test_rejects_foreign_output_and_invalid_limit(self):
        labels.append_labels(self.output, [('ĐÃ CHỐT', 'ham')])
        with self.assertRaisesRegex(ValueError, 'sai thứ tự'):
            labels.main()
        with self.assertRaises(ValueError):
            labels.main(limit=0)
        self.api.assert_not_called()

    def test_rejects_wrong_order_and_detects_missing_output(self):
        labels.append_labels(self.output, [(self.texts[1], 'ham')])
        with self.assertRaisesRegex(ValueError, 'sai thứ tự'):
            labels.main()
        self.output.unlink()
        with patch.object(labels, 'label_chunk'):
            with self.assertRaisesRegex(SystemExit, 'Chưa gán đủ'):
                labels.main()

    def test_rejects_bad_or_duplicate_csv_rows(self):
        for rows in [[('ok', 'unknown')], [(' ', 'ham')], [('ok', 'ham'), ('ok', 'spam')]]:
            with self.subTest(rows=rows):
                with self.output.open('w', encoding='utf-8-sig', newline='') as stream:
                    writer = csv.writer(stream)
                    writer.writerow(['content', 'label'])
                    writer.writerows(rows)
                with self.assertRaises(ValueError):
                    labels.read_labels(self.output)

    def test_parse_spam_rejects_invalid_indices(self):
        self.assertEqual(labels.parse_spam('```json\n{"spam": ["0", 1]}\n```', 2), {0, 1})
        for answer in ['{"spam": [true]}', '{"spam": [-1]}', '{"spam": [2]}', '{"spam": "0"}']:
            with self.subTest(answer=answer), self.assertRaises(ValueError):
                labels.parse_spam(answer, 2)


if __name__ == '__main__':
    unittest.main()
