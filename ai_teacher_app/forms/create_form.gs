/**
 * Google Apps Script: builds a Ms. Nova-compatible quiz Google Form from quiz_config.json.
 * script.google.com > New project > paste this file > paste your config into CONFIG > Run createQuiz.
 * Every question gets a "How sure are you?" item; wrong options must match the config's "distractors".
 * Responses go to a linked Google Sheet; export it as CSV (or share with a service account) for forms_pipeline.py.
 */
const CONFIG = /* paste quiz_config.json here */ {"form_title": "GAI Unit 1 - Week quiz", "questions": []};

function createQuiz() {
  const form = FormApp.create(CONFIG.form_title)
    .setIsQuiz(true)
    .setCollectEmail(true)              // id_column = "Email Address"
    .setShuffleQuestions(false);        // order matters for the learning model
  form.addTextItem().setTitle(CONFIG.name_column || 'Full name').setRequired(true);
  CONFIG.questions.forEach(q => {
    const item = form.addMultipleChoiceItem().setTitle(q.header).setRequired(true);
    const keys = Array.isArray(q.answer) ? q.answer : [q.answer];
    item.setChoices(q.options.map(o => item.createChoice(o, keys.indexOf(o) >= 0))).setPoints(1);
    form.addMultipleChoiceItem().setTitle(q.confidence_header)
        .setChoiceValues(['Sure', 'Not sure', 'Guessing']).setRequired(true);
  });
  const ss = SpreadsheetApp.create(CONFIG.form_title + ' (responses)');
  form.setDestination(FormApp.DestinationType.SPREADSHEET, ss.getId());
  Logger.log('Form:  ' + form.getPublishedUrl());
  Logger.log('Edit:  ' + form.getEditUrl());
  Logger.log('Sheet: ' + ss.getUrl());
}
