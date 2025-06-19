
def brca_prompts():
    prompts = [
        # Labels
        [
            'invasive ductal carcinoma',
            'invasive ductal tumor',
            'invasive ductal',
            'breast ductal carcinoma',
            'breast ductal tumor',
            'ductal carcinoma',
            'ductal tumor',
            'idc',
            ],
        [
            'invasive lobular carcinoma',
            'invasive lobular tumor',
            'invasive lobular',
            'breast lobular carcinoma',
            'breast lobular tumor',
            'lobular carcinoma',
            'lobular tumor',
            'ilc',
            ],
      
                ]
    templates = [
                "CLASSNAME",
                # "a photomicrograph showing CLASSNAME.",
                # "a photomicrograph of CLASSNAME.",
                # "an image of CLASSNAME.",
                # "an image showing CLASSNAME.",
                # "an example of CLASSNAME.",
                # "CLASSNAME is shown.",
                # "this is CLASSNAME.",
                # "there is CLASSNAME.",
                # "a histopathological image showing CLASSNAME.",
                # "a histopathological image of CLASSNAME.",
                # "a histopathological photograph of CLASSNAME.",
                # "a histopathological photograph showing CLASSNAME.",
                # "shows CLASSNAME.",
                # "presence of CLASSNAME.",
                # "CLASSNAME is present.",
                # "an H&E stained image of CLASSNAME.",
                # "an H&E stained image showing CLASSNAME.",
                # "an H&E image showing CLASSNAME.",
                # "an H&E image of CLASSNAME.",
                # "CLASSNAME, H&E stain.",
                # "CLASSNAME, H&E."
            ]

    cls_templates = []
    for i in range(len(prompts)):
        cls_template = []
        for j in range(len(prompts[i])):
            cls_template.extend([template.replace('CLASSNAME', prompts[i][j]) for template in templates])
        cls_templates.append(cls_template)
    return cls_templates

def brca_her2_prompts():
    prompts = [
        # Labels
        [
            'her2 non-expression',
            'her2 non-amplification',
            'her2-negative',
            ],
        [
            'her2 overexpression',
            'her2 amplification',
            'her2-positive',
            ],
       
                ]
    templates = [
                "CLASSNAME",
            ]

    cls_templates = []
    for i in range(len(prompts)):
        cls_template = []
        for j in range(len(prompts[i])):
            cls_template.extend([template.replace('CLASSNAME', prompts[i][j]) for template in templates])
        cls_templates.append(cls_template)
    return cls_templates


def nsclc_prompts():
    prompts = [
        # Labels
        [
            'adenocarcinoma',
            'lung adenocarcinoma',
            'adenocarcinoma of the lung',
            'luad',
            ],
        [
            'squamous cell carcinoma',
            'lung squamous cell carcinoma',
            'squamous cell carcinoma of the lung',
            'lusc',
            ],

       
                ]
    templates = [
                "CLASSNAME",
                # "a photomicrograph showing CLASSNAME.",
                # "a photomicrograph of CLASSNAME.",
                # "an image of CLASSNAME.",
                # "an image showing CLASSNAME.",
                # "an example of CLASSNAME.",
                # "CLASSNAME is shown.",
                # "this is CLASSNAME.",
                # "there is CLASSNAME.",
                # "a histopathological image showing CLASSNAME.",
                # "a histopathological image of CLASSNAME.",
                # "a histopathological photograph of CLASSNAME.",
                # "a histopathological photograph showing CLASSNAME.",
                # "shows CLASSNAME.",
                # "presence of CLASSNAME.",
                # "CLASSNAME is present.",
                # "an H&E stained image of CLASSNAME.",
                # "an H&E stained image showing CLASSNAME.",
                # "an H&E image showing CLASSNAME.",
                # "an H&E image of CLASSNAME.",
                # "CLASSNAME, H&E stain.",
                # "CLASSNAME, H&E."
            ]

    cls_templates = []
    for i in range(len(prompts)):
        cls_template = []
        # cls_templates.append([template.replace('CLASSNAME', prompts[i]) for template in templates])
        for j in range(len(prompts[i])):
            cls_template.extend([template.replace('CLASSNAME', prompts[i][j]) for template in templates])
        cls_templates.append(cls_template)
    # for i in range(len(prompts)):
    #     # cls_templates.append([template.replace('CLASSNAME', prompts[i]) for template in templates])
    #     cls_templates.extend([template.replace('CLASSNAME', prompts[i]) for template in templates])

    return cls_templates


def luad_egfr_prompts():
    prompts = [
        # Labels
        [
            'egfr wildtype',
            'egfr gene wildtype',
            ],
        [
            'egfr mutation',
            'egfr gene mutation',
            ],
        
      
                ]
    templates = [
                "CLASSNAME",
            ]

    cls_templates = []
    for i in range(len(prompts)):
        cls_template = []
        for j in range(len(prompts[i])):
            cls_template.extend([template.replace('CLASSNAME', prompts[i][j]) for template in templates])
        cls_templates.append(cls_template)
    return cls_templates
